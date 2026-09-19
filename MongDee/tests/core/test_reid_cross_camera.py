"""One person seen by two cameras must keep ONE global identity.

Cameras differ in white balance, exposure, sharpness and framing, so the same shirt looks different in each view.
These tests render synthetic people (torso / legs / head blocks), push them through two simulated cameras, and run the
REAL clothing-colour descriptor and the REAL GlobalIdentityRegistry. Only the CNN half of the embedding is simulated
(identity + camera bias + viewpoint), because the sandbox has no network weights."""
from __future__ import annotations

import cv2
import numpy as np

from core.body_cues import COLOR_DESCRIPTOR_DIM, color_descriptor
from core.reid import REID_COLOR_WEIGHT, GlobalIdentityRegistry

# distinct clothing (BGR torso, BGR legs): clearly different people
OUTFITS = [
    ((40, 40, 200), (60, 60, 60)),       # red top, dark trousers
    ((200, 90, 30), (170, 150, 120)),    # blue top, khaki
    ((40, 170, 60), (30, 30, 30)),       # green top, black
    ((30, 200, 220), (150, 60, 40)),     # yellow top, blue jeans
    ((220, 220, 220), (50, 60, 160)),    # white top, brown/red trousers
    ((160, 50, 160), (200, 200, 200)),   # purple top, light grey
]
D = 64


def _person(index: int) -> np.ndarray:
    r = np.random.default_rng(index)
    torso, legs = OUTFITS[index]
    img = np.zeros((192, 96, 3), np.uint8)
    img[:] = r.integers(90, 170, 3)
    img[4:32, 30:66] = (140, 170, 220)                     # face
    img[0:14, 28:68] = (30, 30, 30)                         # hair
    img[36:105, 14:82] = torso
    img[105:182, 24:72] = legs
    return np.clip(img + r.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def _camera(img: np.ndarray, seed: int) -> np.ndarray:
    """A camera with its own white balance, gamma, exposure, sharpness and framing."""
    r = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255
    x = np.clip(x * r.uniform(0.78, 1.25, 3), 0, 1) ** r.uniform(0.7, 1.4) * r.uniform(0.6, 1.3)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    h, w = out.shape[:2]
    s = r.uniform(0.55, 1.0)
    out = cv2.resize(cv2.resize(out, (int(w * s), int(h * s))), (w, h))
    out = cv2.warpAffine(out, np.float32([[1, 0, r.integers(-6, 7)], [0, 1, r.integers(-8, 9)]]), (w, h),
                         borderMode=cv2.BORDER_REPLICATE)
    return np.clip(out + r.normal(0, 6, out.shape), 0, 255).astype(np.uint8)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embedding(cnn: np.ndarray, image: np.ndarray) -> np.ndarray:
    colour = color_descriptor(image)
    n = np.linalg.norm(colour)
    colour = colour / n if n > 0 else colour
    cnn = cnn / np.linalg.norm(cnn)
    e = np.concatenate([np.sqrt(1 - REID_COLOR_WEIGHT) * cnn, np.sqrt(REID_COLOR_WEIGHT) * colour])
    return e / np.linalg.norm(e)


def test_clothing_colour_survives_a_change_of_camera():
    same, diff = [], []
    descs = {(i, c): color_descriptor(_camera(_person(i), 100 * c + i)) for i in range(len(OUTFITS)) for c in (1, 2)}
    for i in range(len(OUTFITS)):
        same.append(_cos(descs[(i, 1)], descs[(i, 2)]))
        diff += [_cos(descs[(i, 1)], descs[(j, 2)]) for j in range(len(OUTFITS)) if j != i]
    assert min(same) > 0.75, same
    assert min(same) > max(diff), (min(same), max(diff))


def _run(overlap: bool, seed: int, **registry_kwargs):
    rng = np.random.default_rng(seed)
    n = len(OUTFITS)
    ident = [rng.normal(size=D) for _ in range(n)]
    cam_bias = {c: rng.normal(size=D) for c in "AB"}
    view = {(i, c): rng.normal(size=D) for i in range(n) for c in "AB"}
    reg = GlobalIdentityRegistry(require_confirmation=True, **registry_kwargs)
    t = 0.0
    for step in range(8):
        t += 0.75
        for i in range(n):
            cams = ["A", "B"] if overlap else (["A"] if i % 2 == 0 else ["B"])
            for c in cams:
                cnn = ident[i] + 0.35 * cam_bias[c] + 0.9 * view[(i, c)] + 0.25 * rng.normal(size=D)
                e = _embedding(cnn, _camera(_person(i), seed * 1000 + i * 31 + ord(c) + step))
                reg.observe_frame(c, [i], t)
                reg.resolve(c, i, e, t, bbox=[100 + i * 40, 50, 140 + i * 40, 200])
    return reg, n


def _score(reg, n):
    """(person-views split over two identities, identities that contain more than one person)"""
    dups = sum(reg.get_global_id_for("A", i) != reg.get_global_id_for("B", i) for i in range(n))
    by: dict = {}
    for i in range(n):
        for c in "AB":
            by.setdefault(reg.get_global_id_for(c, i), set()).add(i)
    return dups, sum(1 for v in by.values() if len(v) > 1)


def test_one_person_in_two_overlapping_views_gets_one_id():
    old_dups = new_dups = old_mixed = new_mixed = 0
    for seed in range(3):
        reg_old, n = _run(True, seed, cross_camera=False)
        reg_new, _ = _run(True, seed, colour_dim=COLOR_DESCRIPTOR_DIM)
        d, m = _score(reg_old, n)
        old_dups, old_mixed = old_dups + d, old_mixed + m
        d, m = _score(reg_new, n)
        new_dups, new_mixed = new_dups + d, new_mixed + m
    assert old_dups >= 4                                   # the problem is real without the fix ...
    # Body look alone is a weak cue between cameras, so the registry is now deliberately strict (it prefers
    # a duplicate ID to gluing two people together); it must still beat the old behaviour and never mix people.
    assert new_dups < old_dups, (old_dups, new_dups)
    assert new_mixed <= old_mixed                          # (the simulator never reports two people on screen together)


def test_cross_camera_logic_does_not_merge_different_people():
    for seed in range(3):
        reg, n = _run(False, seed, colour_dim=COLOR_DESCRIPTOR_DIM)   # each person is seen by ONE camera only
        assert reg.unique_people_count() == n


def test_switching_cross_camera_logic_off_restores_the_old_matching():
    reg, _ = _run(True, 0, cross_camera=False)
    assert reg.stats()["cross_camera_merges"] == 0
    assert reg.stats()["cross_camera_matches"] >= 0


def test_two_people_on_the_same_camera_are_never_merged_across_views():
    reg, n = _run(True, 0, colour_dim=COLOR_DESCRIPTOR_DIM)
    for c in "AB":
        ids = {reg.get_global_id_for(c, i) for i in range(n)}
        assert len(ids) == n                               # six people on one camera -> six identities


# ------------------------------------------------ two DIFFERENT people must never share an ID (field bug, 2026-09-19)
def _scene_like_cnn(rng, scene):
    """The CNN half of the embedding is dominated by the (shared) office background: it says 'similar' for everybody."""
    return scene + 0.15 * rng.normal(size=D)


def _two_strangers(**kw):
    rng = np.random.default_rng(0)
    scene = rng.normal(size=D)
    reg = GlobalIdentityRegistry(require_confirmation=True, **kw)
    t = 0.0
    for step in range(8):
        t += 0.75
        for i, cam in ((0, "A"), (1, "B")):
            e = _embedding(_scene_like_cnn(rng, scene), _camera(_person(i), 500 + step * 7 + i))
            reg.observe_frame(cam, [1], t)
            reg.resolve(cam, 1, e, t, bbox=[100, 50, 200, 300])
    return reg


def test_different_clothes_are_never_one_person_even_if_the_cnn_sees_the_same_office():
    old = _two_strangers(cross_camera=False)
    assert old.get_global_id_for("A", 1) == old.get_global_id_for("B", 1)      # the bug: merged on background alone
    new = _two_strangers(colour_dim=COLOR_DESCRIPTOR_DIM)
    assert new.get_global_id_for("A", 1) != new.get_global_id_for("B", 1)
    assert new.unique_people_count() == 2


def test_different_decided_genders_are_never_matched_or_merged():
    rng = np.random.default_rng(1)
    scene = rng.normal(size=D)
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
    t = 0.0
    for step in range(8):
        t += 0.75
        for cam in "AB":                                    # the SAME outfit on both cameras - look alike strangers
            e = _embedding(_scene_like_cnn(rng, scene), _camera(_person(0), 900 + step * 3 + ord(cam)))
            reg.observe_frame(cam, [1], t)
            reg.resolve(cam, 1, e, t, bbox=[100, 50, 200, 300], gender="female" if cam == "A" else "male")
            reg.set_person_gender(reg.get_global_id_for(cam, 1), "female" if cam == "A" else "male")
    assert reg.get_global_id_for("A", 1) != reg.get_global_id_for("B", 1)


def test_a_track_wrongly_mapped_to_someone_else_is_split_off_and_never_mapped_back():
    rng = np.random.default_rng(2)
    scene = rng.normal(size=D)
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM, merging=False)
    t = 0.0
    for step in range(4):                                   # the woman on camera A
        t += 0.75
        reg.observe_frame("A", [1], t)
        reg.resolve("A", 1, _embedding(_scene_like_cnn(rng, scene), _camera(_person(0), 40 + step)), t, bbox=[100, 50, 200, 300])
    woman = reg.get_global_id_for("A", 1)
    reg.set_person_gender(woman, "female")
    # a man on camera B is (wrongly) attached to her identity, as an old build could do
    with reg._lock:
        reg._local_to_global[("B", 1)] = woman
        reg._people[woman].add_embedding(_embedding(_scene_like_cnn(rng, scene), _camera(_person(1), 3)), "B", ("B", 1))
    for step in range(4):
        t += 0.75
        reg.observe_frame("B", [1], t)
        gid, _ = reg.resolve("B", 1, _embedding(_scene_like_cnn(rng, scene), _camera(_person(1), 70 + step)), t,
                             bbox=[100, 50, 200, 300], gender="male")
    assert gid != woman
    assert reg.get_global_id_for("B", 1) != woman and reg.get_global_id_for("A", 1) == woman
    assert reg.stats()["split_wrong_identity"] >= 1
    assert not any(k is not None and k == ("B", 1) for k in reg.get_person(woman).keys_of_embeddings())


def test_same_person_with_a_colour_cast_is_not_split():
    reg, n = _run(True, 0, colour_dim=COLOR_DESCRIPTOR_DIM)
    assert reg.stats()["split_wrong_identity"] == 0


# ------------------------------------------- face identity (core.face_identity) + strict body-only cross-camera rules
def _unit(v):
    return v / np.linalg.norm(v)


def _face(base, rng, noise=1.0):
    """A noisy view of one face: cosine to the true face ~0.7 (real same-person SFace scores are 0.4 - 0.8)."""
    return _unit(base + noise * rng.normal(size=base.shape) / np.sqrt(len(base)))


def _run_faces(seed: int, p_face: float, overlap: bool = True):
    rng = np.random.default_rng(seed)
    n = len(OUTFITS)
    ident = [rng.normal(size=D) for _ in range(n)]
    faces = [_unit(rng.normal(size=128)) for _ in range(n)]
    cam_bias = {c: rng.normal(size=D) for c in "AB"}
    view = {(i, c): rng.normal(size=D) for i in range(n) for c in "AB"}
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
    t = 0.0
    for step in range(12):
        t += 0.75
        for i in range(n):
            for c in ("A", "B") if overlap else (("A",) if i % 2 == 0 else ("B",)):
                cnn = ident[i] + 0.35 * cam_bias[c] + 0.9 * view[(i, c)] + 0.25 * rng.normal(size=D)
                e = _embedding(cnn, _camera(_person(i), seed * 1000 + i * 31 + ord(c) + step))
                f = _face(faces[i], rng) if rng.random() < p_face else None
                reg.observe_frame(c, [i], t)
                reg.resolve(c, i, e, t, bbox=[100 + i * 40, 50, 140 + i * 40, 200], face=f)
    return reg, n


def test_face_embeddings_join_one_person_across_two_cameras():
    dups = mixed = 0
    for seed in range(3):
        reg, n = _run_faces(seed, p_face=0.5)
        d, m = _score(reg, n)
        dups, mixed = dups + d, mixed + m
    assert mixed == 0
    assert dups <= 1, dups                                  # 18 person-views


def test_different_faces_keep_two_people_apart_even_if_the_body_looks_identical():
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
    rng = np.random.default_rng(3)
    body = _embedding(rng.normal(size=D), _person(0))       # exactly the same body look on both cameras (and stable)
    woman, man = _unit(rng.normal(size=128)), _unit(rng.normal(size=128))
    t = 0.0
    for _ in range(10):
        t += 0.75
        reg.observe_frame("A", [1], t)
        reg.resolve("A", 1, body, t, bbox=[100, 50, 140, 200], face=_face(woman, rng))
        reg.observe_frame("B", [1], t)
        reg.resolve("B", 1, body, t, bbox=[100, 50, 140, 200], face=_face(man, rng))
    assert reg.get_global_id_for("A", 1) != reg.get_global_id_for("B", 1)
    assert reg.unique_people_count() == 2


def test_the_same_face_links_two_cameras_even_when_the_clothes_look_different():
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
    rng = np.random.default_rng(4)
    face = _unit(rng.normal(size=128))
    cnn_a, cnn_b = rng.normal(size=D), rng.normal(size=D)
    t = 0.0
    for step in range(10):
        t += 0.75
        ea = _embedding(cnn_a + 0.1 * rng.normal(size=D), _camera(_person(0), step))
        eb = _embedding(cnn_b + 0.1 * rng.normal(size=D), _camera(_person(3), 50 + step))   # different clothes + look
        reg.observe_frame("A", [1], t)
        reg.resolve("A", 1, ea, t, bbox=[100, 50, 140, 200], face=_face(face, rng))
        reg.observe_frame("B", [7], t)
        reg.resolve("B", 7, eb, t, bbox=[300, 50, 340, 200], face=_face(face, rng))
    assert reg.get_global_id_for("A", 1) == reg.get_global_id_for("B", 7)
    assert reg.unique_people_count() == 1


def _emb_with_colour_agreement(cnn, colour):
    e = np.concatenate([np.sqrt(0.7) * _unit(cnn), np.sqrt(0.3) * _unit(colour)])
    return e / np.linalg.norm(e)


def test_a_woman_and_a_man_with_0_57_clothing_agreement_are_two_people():
    """Field bug: a real booth crop pair had colour agreement 0.57 and a background-dominated CNN similarity of ~0.9;
    the old cross-camera rules glued them into one ID and the man was labelled FEMALE."""
    rng = np.random.default_rng(11)
    bg = rng.normal(size=D)                                  # the shared booth background dominates the CNN feature
    c1 = rng.normal(size=COLOR_DESCRIPTOR_DIM)
    orth = rng.normal(size=COLOR_DESCRIPTOR_DIM)
    orth -= orth @ _unit(c1) * _unit(c1)
    c2 = 0.57 * _unit(c1) + np.sqrt(1 - 0.57 ** 2) * _unit(orth)
    reg = GlobalIdentityRegistry(require_confirmation=True, colour_dim=COLOR_DESCRIPTOR_DIM)
    t = 0.0
    for _ in range(12):
        t += 0.75
        ew = _emb_with_colour_agreement(bg + 0.25 * rng.normal(size=D), c1 + 0.05 * rng.normal(size=c1.shape))
        em = _emb_with_colour_agreement(bg + 0.25 * rng.normal(size=D), c2 + 0.05 * rng.normal(size=c2.shape))
        reg.observe_frame("A", [1], t)
        reg.resolve("A", 1, ew, t, bbox=[100, 50, 140, 200])
        reg.observe_frame("B", [1], t)
        reg.resolve("B", 1, em, t, bbox=[100, 50, 140, 200])
    assert reg.get_global_id_for("A", 1) != reg.get_global_id_for("B", 1)


def _hair(dark=0.9, width=0.5, side=0.05, below=0.02, sat=0.2, val=0.2):
    return np.array([dark, width, side, below, sat, val], np.float32)


def _two_camera_registry(hair_a, hair_b, same_clothes=True):
    reg = GlobalIdentityRegistry(require_confirmation=False, colour_dim=COLOR_DESCRIPTOR_DIM)
    rng = np.random.default_rng(5)
    cnn = rng.normal(size=D)
    ea = _embedding(cnn, _person(1))
    eb = ea if same_clothes else _embedding(cnn, _person(4))
    reg.observe_frame("A", [1], 1.0)
    gid_a, _ = reg.resolve("A", 1, ea, 1.0, bbox=[100, 50, 140, 200], hair=hair_a)
    reg.forget_local_track("A", 1)
    gid_b, _ = reg.resolve("B", 9, eb, 2.0, bbox=[100, 50, 140, 200], hair=hair_b)
    return gid_a, gid_b


def test_body_link_across_cameras_needs_every_cue_to_agree():
    a, b = _two_camera_registry(_hair(), _hair(dark=0.88))              # same clothes, same hairstyle
    assert a == b
    a, b = _two_camera_registry(_hair(), _hair(below=0.55, side=0.5))   # same clothes, but long hair vs short hair
    assert a != b
    a, b = _two_camera_registry(_hair(), _hair(), same_clothes=False)   # same hair, different clothes
    assert a != b


def test_cannot_confirm_means_a_separate_id():
    a, b = _two_camera_registry(_hair(), _hair(dark=0.2, val=0.8, sat=0.7))   # blonde / bare-headed vs black hair
    assert a != b
