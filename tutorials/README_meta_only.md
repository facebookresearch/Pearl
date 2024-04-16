Make sure to reflect any changes on tutorials to the correposponding
unit tests in test/unit/test_tutorials,
so that we are alerted to code changes breaking the tutorial.
Likewise, make sure to reflect any changes in the tests to the tutorial.

Note that this synchronization is not a copy-and-paste affair;
one must be careful to replace things as needed:
- the notebook uses `plt.show()` to show the plot, the unit test
does not compute the graphs (the code is kept commented out for reference)
- the paths to the files (such as .pt files) must use "pearl" in fbcode and "Pearl" in
  the open-source version.
