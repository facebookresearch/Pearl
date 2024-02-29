# Online storage for Pearl testing and demoing files

Pearl tests and demos sometimes need to use data files. Storing them directly in the repository would penalize users cloning it who do not need it. Instead, we would like to have a reliable, permanent website under our control from which to download them, in a way that is accessible to code running either inside or outside Meta.

We have created a GitHub Pages branch gh-pages in our GitHub Pearl repository that can do just that. It can be accessed at https://github.com/facebookresearch/Pearl/tree/gh-pages. It contains a data directory where files can be stored (directly in the browser) and then accessed through https://facebookresearch.github.io/Pearl/data.

Typically, GitHub Pages is used for storing a project's web site. Because we already have the official Pearl web site at https://pearlagent.github.io/, the Pearl GitHub Pages home page simply redirects to that. In the future we might want to move the official site to the Pearl repository GitHub Pages in order to have everything under a single repository, though.

Note that this branch is not synchronized with the Meta internal version of Pearl.
