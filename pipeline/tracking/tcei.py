"""TCEI calibration around unmodified MOTIP; see docs/tcei.md for deviations.

Reference: 1941Zpf/TCEI, commit 145d1b8 (CVPR 2026). Only the reference
entropy/fusion helpers are imported. Object features come from raw DETR output;
ID features come from the final ID classification head's input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TceiConfig:
    mode: str = "dual"
    cache_scope: str = "sequence"
    implementation: str = "motip-hooks-corrected-v1"
    pos_capacity: int = 3
    neg_capacity: int = 2
    object_pos_alpha: float = 1.0
    object_pos_beta: float = 3.0
    object_neg_alpha: float = 0.117
    object_neg_beta: float = 1.5
    object_temperature: float = 0.05
    id_pos_alpha: float = 1.0
    id_pos_beta: float = 5.0
    id_neg_alpha: float = 0.117
    id_neg_beta: float = 2.0
    entropy_lower: float = 0.2
    entropy_upper: float = 0.5
    mask_lower: float = 0.03
    mask_upper: float = 1.0
    lambda_id: float = 1.0

    def __post_init__(self):
        if self.mode not in {"off", "object", "id", "dual"}:
            raise ValueError(self.mode)
        if self.cache_scope not in {"sequence", "stream"}:
            raise ValueError(self.cache_scope)
        if min(self.pos_capacity, self.neg_capacity) < 1:
            raise ValueError("Cache capacities must be positive")


class Calibrator:
    def __init__(self, config, reference, wide_caches=None):
        self.config, self.reference = config, reference
        self.local = ({}, {})
        self.wide = ({}, {}) if wide_caches is None else wide_caches
        self.calibrated_rows = 0
        self.last_logits = None

    @staticmethod
    def insert(cache, key, features, loss, capacity, negatives=None):
        items = cache.setdefault(int(key), [])
        items.append((features.detach().clone(), float(loss), negatives))
        items.sort(key=lambda entry: entry[1])
        del items[capacity:]

    def seed(self, identity, features, loss):
        self.insert(
            self.local[0],
            identity,
            F.normalize(features.float(), dim=-1),
            loss,
            self.config.pos_capacity,
        )

    def prune(self, active):
        for cache in self.local:
            for identity in set(cache) - set(active):
                del cache[identity]

    def cache_logits(self, features, cache, classes, mapping, *, wide, negative):
        if not cache:
            return features.new_zeros((1, classes))
        cfg = self.config
        values, similarities = [], []
        reverse = {identity: slot for slot, identity in mapping.items()}
        for key in sorted(cache):
            entries = cache[key]
            if wide:
                for feature, _, negatives in entries:
                    similarities.append((features @ feature.T).squeeze())
                    if negative:
                        values.append(
                            (
                                (negatives > cfg.mask_lower)
                                & (negatives < cfg.mask_upper)
                            ).float()[0]
                        )
                    else:
                        values.append(
                            F.one_hot(
                                torch.tensor(key, device=features.device), classes
                            ).float()
                        )
            else:
                # Author local cache averages features within each actual track.
                similarities.append(
                    torch.cat(
                        [features @ entry[0].T for entry in entries], dim=-1
                    ).mean()
                )
                value = features.new_zeros(classes)
                targets = entries[-1][2] if negative else (key,)
                for identity in targets:
                    slot = reverse.get(identity)
                    if slot is not None:
                        value[slot] = 1
                values.append(value)
        affinity = torch.stack(similarities)[None]
        if not wide:
            # Equal/single-class similarities have no relative information.
            spread = affinity.max() - affinity.min()
            affinity = (affinity - affinity.min()) / spread.clamp_min(1e-8)
            if cfg.object_temperature:
                affinity = (affinity / cfg.object_temperature).softmax(dim=-1)
        alpha = (
            (cfg.id_neg_alpha if negative else cfg.id_pos_alpha)
            if wide
            else (cfg.object_neg_alpha if negative else cfg.object_pos_alpha)
        )
        beta = (
            (cfg.id_neg_beta if negative else cfg.id_pos_beta)
            if wide
            else (cfg.object_neg_beta if negative else cfg.object_pos_beta)
        )
        return alpha * torch.exp(-beta * (1 - affinity)) @ torch.stack(values)

    @torch.no_grad()
    def __call__(self, logits, object_features, id_features, labels, mapping):
        cfg = self.config
        if cfg.mode == "off":
            return logits
        if logits.dtype != torch.float32:
            raise ValueError("TCEI experiments require FP32")
        classes = logits.shape[-1]
        newborn = classes - 1
        result = logits.clone()
        for index, slot in enumerate(labels):
            row = logits[index : index + 1]
            loss, probability, _ = self.reference.logits_process(row)
            # Preserve author's entropy normalization (natural log / log2(K)).
            entropy = self.reference.get_entropy(loss, classes)
            uncertain = cfg.entropy_lower < entropy < cfg.entropy_upper
            id_delta = torch.zeros_like(row)
            if cfg.mode in {"id", "dual"}:
                feature = F.normalize(id_features[index : index + 1].float(), dim=-1)
                self.insert(self.wide[0], slot, feature, loss, cfg.pos_capacity)
                if uncertain:
                    self.insert(
                        self.wide[1],
                        slot,
                        feature,
                        loss,
                        cfg.neg_capacity,
                        probability.detach().clone(),
                    )
                id_delta = self.cache_logits(
                    feature, self.wide[0], classes, mapping, wide=True, negative=False
                )
                id_delta -= self.cache_logits(
                    feature, self.wide[1], classes, mapping, wide=True, negative=True
                )
            # Author collects wide examples for newborns but does not calibrate them.
            if slot == newborn:
                continue
            object_delta = torch.zeros_like(row)
            if cfg.mode in {"object", "dual"}:
                feature = F.normalize(
                    object_features[index : index + 1].float(), dim=-1
                )
                identity = mapping[slot]
                self.insert(self.local[0], identity, feature, loss, cfg.pos_capacity)
                if uncertain:
                    targets = tuple(
                        actual
                        for token, actual in mapping.items()
                        if cfg.mask_lower
                        < float(probability[0, token])
                        < cfg.mask_upper
                    )
                    self.insert(
                        self.local[1],
                        identity,
                        feature,
                        loss,
                        cfg.neg_capacity,
                        targets,
                    )
                object_delta = self.cache_logits(
                    feature, self.local[0], classes, mapping, wide=False, negative=False
                )
                object_delta -= self.cache_logits(
                    feature, self.local[1], classes, mapping, wide=False, negative=True
                )
            if cfg.mode == "dual":
                perpendicular = self.reference.fisher_orthogonalize(
                    object_delta, id_delta, probability
                )
                result[index] = (
                    row
                    + object_delta
                    + cfg.lambda_id * perpendicular * probability * (1 - probability)
                )
            else:
                result[index] = row + (
                    object_delta if cfg.mode == "object" else id_delta
                )
            self.calibrated_rows += 1
        if not torch.isfinite(result).all():
            raise FloatingPointError("TCEI produced nonfinite logits")
        self.last_logits = result.detach().clone()
        return result


class _TrackerMixin:
    def _get_id_pred_labels(self, boxes, output_embeds):
        self._pre_filter_features = output_embeds.detach()
        self.calibrator.last_logits = None
        if (
            self.calibrator.config.mode == "off"
            or self.trajectory_features.shape[0] == 0
        ):
            return super()._get_id_pred_labels(boxes, output_embeds)
        captured = []

        def capture_head(module, inputs):
            # Shared head is called at every decoder layer: retain the final one.
            captured[:] = [inputs[0].detach()]

        def calibrate(module, inputs, output):
            logits = output[0]
            raw = logits[0, 0, 0]
            labels = self._object_max_assignment(id_scores=raw.softmax(-1))
            features = captured[-1].reshape(-1, captured[-1].shape[-1])
            calibrated = self.calibrator(
                raw, output_embeds, features, labels, self.id_label_to_id
            )
            return (calibrated.reshape_as(logits), *output[1:])

        decoder = self.model.id_decoder
        head_handle = decoder.embed_to_word_layers[-1].register_forward_pre_hook(
            capture_head
        )
        decoder_handle = decoder.register_forward_hook(calibrate)
        try:
            return super()._get_id_pred_labels(boxes, output_embeds)
        finally:
            decoder_handle.remove()
            head_handle.remove()

    def _assign_newborn_id_labels(self, pred_id_labels):
        before = self.next_id
        labels = super()._assign_newborn_id_labels(pred_id_labels)
        self._newborn_actual_ids = set(range(before, self.next_id))
        return labels

    def _update_trajectory_infos(self, boxes, output_embeds, id_labels):
        if self.calibrator.config.mode in {"object", "dual"}:
            for index, slot in enumerate(id_labels.tolist()):
                identity = self.id_label_to_id[slot]
                if identity in self._newborn_actual_ids:
                    # Author indexes pre-filter logits with post-filter positions.
                    # Find the retained raw feature to preserve the right entropy.
                    feature = output_embeds[index : index + 1]
                    loss = math.log(self.num_id_vocabulary)
                    if self.calibrator.last_logits is not None:
                        matches = torch.where(
                            (self._pre_filter_features == feature).all(dim=-1)
                        )[0]
                        loss, _, _ = self.calibrator.reference.logits_process(
                            self.calibrator.last_logits[matches[:1]]
                        )
                    self.calibrator.seed(identity, feature, loss)
        return super()._update_trajectory_infos(boxes, output_embeds, id_labels)

    def _filter_out_inactive_tracks(self):
        super()._filter_out_inactive_tracks()
        active = (
            self.trajectory_id_labels[0].tolist()
            if self.trajectory_id_labels.shape[0]
            else []
        )
        self.calibrator.prune({self.id_label_to_id[slot] for slot in active})


def create_tracker(base, *, calibration, reference, wide_caches, **kwargs):
    if (
        kwargs["assignment_protocol"] != "object-max"
        or kwargs["use_sigmoid"]
        or kwargs["dtype"] != torch.float32
        or kwargs["only_detr"]
    ):
        raise ValueError("TCEI supports the audited FP32/softmax/object-max protocol")
    tracker_class = type("TceiRuntimeTracker", (_TrackerMixin, base), {})
    tracker = tracker_class(**kwargs)
    tracker.calibrator = Calibrator(calibration, reference, wide_caches)
    return tracker
