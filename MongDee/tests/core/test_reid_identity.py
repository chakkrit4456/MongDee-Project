"""Re-ID identity stability: box flicker / track churn must not mint new IDs, and one person must not
be counted twice. Embeddings are synthetic unit vectors with a controlled cosine to a base look."""
import numpy as np

from core.reid import GlobalIdentityRegistry

DIM = 64


def _unit(v):
    return v / np.linalg.norm(v)


_rng = np.random.default_rng(3)
LOOK_A = _unit(_rng.normal(size=DIM))
LOOK_B = _unit(_rng.normal(size=DIM))
ORTHO_A = _unit(_rng.normal(size=DIM) - LOOK_A * 0)   # a direction to mix in; near-orthogonal in 64-d
ORTHO_A = _unit(ORTHO_A - LOOK_A * float(ORTHO_A @ LOOK_A))


def look(base, cos, ortho=None, seed=0):
    """A unit vector whose cosine with `base` is `cos` (a person seen from another angle / lighting)."""
    o = ortho if ortho is not None else _unit(np.random.default_rng(seed).normal(size=DIM))
    o = _unit(o - base * float(o @ base))
    return _unit(cos * base + np.sqrt(1 - cos ** 2) * o)


BOX = [100.0, 40.0, 160.0, 200.0]


def moved(dx):
    return [BOX[0] + dx, BOX[1], BOX[2] + dx, BOX[3]]


def test_track_churn_after_a_flicker_keeps_the_same_identity_when_place_and_time_agree():
    reg = GlobalIdentityRegistry()
    gid, _ = reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.0, bbox=BOX)
    reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=2), 0.8, bbox=BOX)
    reg.forget_local_track("CAM-1", 1)                      # tracker lost the person for a moment
    # Reborn 1.5 s later as track 2, seen from a different angle: the look alone (cos 0.35) would NOT match.
    gid2, is_new = reg.resolve("CAM-1", 2, look(LOOK_A, 0.35, seed=9), 2.3, bbox=moved(15))
    assert gid2 == gid and not is_new
    assert reg.stats()["stitched"] == 1 and reg.unique_people_count() == 1


def test_without_stitching_the_same_flicker_mints_a_second_identity():
    reg = GlobalIdentityRegistry(stitching=False)
    gid, _ = reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.0, bbox=BOX)
    reg.forget_local_track("CAM-1", 1)
    gid2, is_new = reg.resolve("CAM-1", 2, look(LOOK_A, 0.35, seed=9), 2.3, bbox=moved(15))
    assert gid2 != gid and is_new


def test_a_different_place_or_a_long_gap_is_not_stitched():
    reg = GlobalIdentityRegistry()
    gid, _ = reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.0, bbox=BOX)
    reg.forget_local_track("CAM-1", 1)
    far, _ = reg.resolve("CAM-1", 2, look(LOOK_A, 0.35, seed=9), 2.0, bbox=moved(600))
    assert far != gid
    reg2 = GlobalIdentityRegistry()
    g1, _ = reg2.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.0, bbox=BOX)
    reg2.forget_local_track("CAM-1", 1)
    late, _ = reg2.resolve("CAM-1", 2, look(LOOK_A, 0.35, seed=9), 30.0, bbox=BOX)
    assert late != g1


def test_stitching_still_needs_a_sane_appearance():
    reg = GlobalIdentityRegistry()
    gid, _ = reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.0, bbox=BOX)
    reg.forget_local_track("CAM-1", 1)
    other, _ = reg.resolve("CAM-1", 2, LOOK_B, 1.5, bbox=BOX)       # someone completely different, same spot
    assert other != gid


def test_two_people_on_screen_together_are_never_the_same_identity():
    reg = GlobalIdentityRegistry()
    a, _ = reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    reg.observe_frame("CAM-1", [1], 0.0)
    # A second track appears while person A is still on screen and looks alike (same uniform): different person.
    b, is_new = reg.resolve("CAM-1", 2, look(LOOK_A, 0.9, seed=4), 0.2, bbox=moved(200))
    assert b != a and is_new


def test_two_people_who_walk_in_together_never_share_an_id():
    """Field bug: two different real people, BOTH brand new, resolved for the first time in the exact same
    AI pass (e.g. two people who walked in together) — the ordinary web/booth_manager.py call order is
    observe_frame() ONCE at the top of the pass (before either track has a global_id mapping yet, so it
    sees neither of them) followed by resolve() for each visible track. Without _mark_on_screen widening
    the on-screen set as each track resolves, the second track's match loop still sees the ZERO people
    observe_frame() reported and so is never excluded from matching the first track's brand-new identity —
    two different people could be minted onto ONE Global ID the moment they meet the similarity floor,
    which does not require them to look alike, just not-too-different (a plausible real scenario: same
    uniform, similar jacket colour, etc.)."""
    reg = GlobalIdentityRegistry()
    reg.observe_frame("CAM-1", [1, 2], 0.0)          # both track ids visible; NEITHER mapped yet (first pass)
    a, is_new_a = reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    b, is_new_b = reg.resolve("CAM-1", 2, look(LOOK_A, 0.9, seed=4), 0.0, bbox=moved(200))
    assert a != b and is_new_a and is_new_b
    # ... and a later merge pass must not fold them back into one person either (coexistence recorded).
    pa, pb = reg.get_person(a), reg.get_person(b)
    assert b in pa.coexisted and a in pb.coexisted


def test_unique_count_ignores_one_blink_ghosts_until_confirmed():
    reg = GlobalIdentityRegistry(require_confirmation=True)
    reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    assert reg.unique_people_count() == 0                    # a single observation is not a person yet
    reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=1), 0.8, bbox=BOX)
    assert reg.unique_people_count() == 1
    assert reg.get_person("P000001").confirmed


def test_split_identities_of_one_person_are_merged_and_counted_once():
    reg = GlobalIdentityRegistry(require_confirmation=True, stitching=False)
    a, _ = reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    reg.resolve("CAM-1", 1, look(LOOK_A, 0.97, seed=1), 1.0, bbox=BOX)
    reg.forget_local_track("CAM-1", 1)
    # Later the same person is minted again from a poor first crop (cos 0.30) ...
    b, is_new = reg.resolve("CAM-1", 7, look(LOOK_A, 0.30, seed=2), 20.0, bbox=moved(300))
    assert b != a and is_new
    # ... and then keeps showing their normal look: the two identities are recognised as one.
    final = b
    for k in range(4):
        final, _ = reg.resolve("CAM-1", 7, look(LOOK_A, 0.93, seed=10 + k), 21.0 + 1.0 * k, bbox=moved(300))
    assert final == a
    assert reg.unique_people_count() == 1
    assert reg.pop_merges() == [(b, a)]
    assert reg.get_global_id_for("CAM-1", 7) == a
    assert reg.pop_merges() == []


def test_identities_seen_on_screen_together_are_never_merged():
    reg = GlobalIdentityRegistry(require_confirmation=True, stitching=False)
    a, _ = reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    b, _ = reg.resolve("CAM-1", 2, look(LOOK_A, 0.30, seed=2), 0.1, bbox=moved(300))
    reg.observe_frame("CAM-1", [1, 2], 0.2)                  # both visible in the same pass
    for k in range(6):
        reg.resolve("CAM-1", 1, look(LOOK_A, 0.97, seed=20 + k), 1.0 + k, bbox=BOX)
        reg.resolve("CAM-1", 2, look(LOOK_A, 0.93, seed=30 + k), 1.0 + k, bbox=moved(300))
    assert reg.pop_merges() == [] and reg.unique_people_count() == 2


def test_tracker_id_swap_is_relinked_after_a_few_disagreeing_samples():
    reg = GlobalIdentityRegistry(stitching=False, merging=False)
    a, _ = reg.resolve("CAM-1", 1, LOOK_A, 0.0, bbox=BOX)
    for k in range(3):
        reg.resolve("CAM-1", 1, look(LOOK_A, 0.95, seed=k), 1.0 + k, bbox=BOX)
    # The tracker now follows somebody else under the same local id.
    ids = [reg.resolve("CAM-1", 1, look(LOOK_B, 0.97, seed=40 + k), 5.0 + k, bbox=BOX)[0] for k in range(4)]
    assert ids[0] == a                                          # tolerated at first (one bad crop)
    assert ids[-1] != a and reg.stats()["relinked"] == 1


def test_defaults_still_behave_like_before_for_a_single_camera_and_person():
    reg = GlobalIdentityRegistry()
    gid, is_new = reg.resolve("CAM-1", 1, LOOK_A, 0.0)
    assert is_new and reg.unique_people_count() == 1
    again, is_new = reg.resolve("CAM-1", 1, look(LOOK_A, 0.9, seed=1), 0.5)
    assert again == gid and not is_new
