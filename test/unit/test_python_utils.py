# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

from dataclasses import dataclass

from pearl.utils.python_utils import get_subdataclass_specific_attributes


# Class definitions for testing


class NonDataclass:
    pass


@dataclass
class DataclassDerivedFromNonDataclass(NonDataclass):
    field: int


@dataclass
class Base:
    common1: int
    common2: int


@dataclass
class Intermediate(Base):
    intermediate_field: float


@dataclass
class DerivedSingle(Base):
    specific: str


@dataclass
class DerivedNone(Base):
    pass


@dataclass
class MultiBase1:
    pass


@dataclass
class MultiBase2:
    common: int


@dataclass
class DerivedMulti(DerivedSingle, MultiBase2):
    another: float


@dataclass
class DerivedMultiple(Base):
    specific1: str
    specific2: float


@dataclass
class DerivedDeep(Intermediate):
    deep_specific: str


@dataclass
class DerivedOverride(Base):
    common1: int


# Test class
class TestGetSubclassSpecificAttributes(unittest.TestCase):
    def test_valid_single_superclass(self) -> None:
        self.assertEqual(
            get_subdataclass_specific_attributes(DerivedSingle), ["specific"]
        )

    def test_no_superclass(self) -> None:
        with self.assertRaises(ValueError):
            get_subdataclass_specific_attributes(Base)

    def test_multiple_superclasses(self) -> None:
        with self.assertRaises(ValueError):
            get_subdataclass_specific_attributes(DerivedMulti)

    def test_non_dataclass_input(self) -> None:
        with self.assertRaises(ValueError):
            get_subdataclass_specific_attributes(NonDataclass)

    def test_superclass_not_dataclass(self) -> None:
        with self.assertRaises(ValueError):
            get_subdataclass_specific_attributes(DataclassDerivedFromNonDataclass)

    def test_no_specific_fields(self) -> None:
        self.assertEqual(get_subdataclass_specific_attributes(DerivedNone), [])

    def test_multiple_specific_fields(self) -> None:
        expected_fields = ["specific1", "specific2"]
        result = get_subdataclass_specific_attributes(DerivedMultiple)
        self.assertEqual(set(result), set(expected_fields))

    def test_inherited_fields_only(self) -> None:
        self.assertEqual(
            get_subdataclass_specific_attributes(Intermediate), ["intermediate_field"]
        )

    def test_deep_inheritance(self) -> None:
        self.assertEqual(
            get_subdataclass_specific_attributes(DerivedDeep), ["deep_specific"]
        )

    def test_overridden_fields(self) -> None:
        self.assertEqual(get_subdataclass_specific_attributes(DerivedOverride), [])


if __name__ == "__main__":
    unittest.main()
