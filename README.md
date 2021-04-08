# Attention is all you need

```python
import torch
from transformer.transformer import Transformer


src = torch.randint(0, 256, (1, 1024))
tgt = torch.randint(0, 256, (1, 1024))

T = Transformer(256,412)
ans = T(src, tgt)
```



## Citations

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need},
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

