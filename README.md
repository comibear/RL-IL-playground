# Basic Implementation for RL & IL

willing to implement in scratch

## Automation

Install the [pre-commit](https://pre-commit.com/) hooks once to keep the
algorithms list below in sync with the repository structure:

```bash
pip install pre-commit
pre-commit install
```

Every `git commit` or `git push` will now refresh the tree so the README always
shows the current implementation status.

## Implemented algorithms
<!-- algorithms-tree start -->
```
RL/algorithms/
├── policy-based/
│   ├── 1.REINFORCE/
│   │   └── REINFORCE-discrete.py
│   └── 2.DDPG/
│       └── ddpg.py
└── value-based/
    └── ddqn/
        └── ddqn.py
```
<!-- algorithms-tree end -->


