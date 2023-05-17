from collections import namedtuple


"""
Transition is designed for one single set of data
"""
Transition = namedtuple(
    "Transition",
    (
        "state",
        "action",
        "reward",
        "next_state",
        "next_available_actions",
        "next_available_actions_mask",
        "done",
    ),
)

"""
TransitionBatch is designed for data batch
"""
TransitionBatch = namedtuple(
    "TransitionBatch",
    (
        "state",
        "action",
        "reward",
        "next_state",
        "next_available_actions",
        "next_available_actions_mask",
        "done",
    ),
)
