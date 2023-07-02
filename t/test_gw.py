import pytest

from mytf.grid_world import GridWorld, Room


@pytest.fixture
def gw():
    room = Room(5, 5)
    room[4, 4] = Room(6, 6)
    room.cellify_partitions()
    room.strip_useless_walls()
    room.condense()

    gw = GridWorld(room=room)
    if gw.T.location.pos != (2, 2):
        gw.R[2, 2] = gw.T  # Move the Turtle to one room
    if gw.G.location.pos != (7, 9):
        gw.R[7, 9] = gw.G  # and move the Goal to the other
    gw.save_initial()  # make sure this is the reset state

    return gw


def test_gw(gw):
    assert gw.T.location.pos == (2, 2)
