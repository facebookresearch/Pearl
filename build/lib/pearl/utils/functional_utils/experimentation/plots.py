# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#


def fontsize_for(
    string: str,
    max_fontsize: float = 24,
    max_number_of_characters_at_max_fontsize: int = 32,
) -> float:
    """
    Computes the font size for a string to fit in an area with given maximum font size and
    maximum number of characters at maximum font size.

    Args:
        string (str): the string to compute the font size for.
        max_fontsize (float, optional): the maximum font size. Defaults to 24.
        max_number_of_characters_at_max_fontsize (int, optional): the maximum number of characters
                            at maximum font size. Defaults to 32.

    Returns:
        _type_: _description_
    """
    total_number_of_title_points = (
        max_number_of_characters_at_max_fontsize * max_fontsize
    )
    number_of_characters = len(string)
    fontsize = total_number_of_title_points // number_of_characters
    return fontsize
