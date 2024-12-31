# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import fields, is_dataclass


def get_subdataclass_specific_attributes(a_dataclass: type[object]) -> list[str]:
    """
    Assumes argument is a dataclass with a single superclass which is also a dataclass.
    Returns a list of attribute names that are specific to the subclass
    (i.e. not present in the superclass).
    """
    # Ensure the class is a dataclass and has a single superclass
    if not is_dataclass(a_dataclass):
        raise ValueError(f"The provided class {a_dataclass} must be a dataclass.")

    if len(a_dataclass.__bases__) != 1:
        raise ValueError(
            f"The provided class must be a dataclass with a single superclass but has bases "
            f"{', '.join(map(str, a_dataclass.__bases__))}"
        )

    if not is_dataclass(a_dataclass) or len(a_dataclass.__bases__) != 1:
        raise ValueError(
            "The provided class must be a dataclass with a single superclass."
        )

    # Get the superclass
    base_class = a_dataclass.__bases__[0]

    # Ensure the superclass is also a dataclass
    if not is_dataclass(base_class):
        raise ValueError("The superclass must also be a dataclass.")

    # Get fields for both the subclass (a_dataclass) and the superclass
    subclass_fields = {f.name for f in fields(a_dataclass)}
    base_class_fields = {f.name for f in fields(base_class)}

    # Return the difference between the subclass fields and superclass fields
    specific_fields = subclass_fields - base_class_fields
    return list(specific_fields)
