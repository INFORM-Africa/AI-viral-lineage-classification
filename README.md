# AI viral lineage classification

This repository hosts two related tools for viral sequence analysis and lineage classification:

| Project | Description | Documentation |
|---------|-------------|---------------|
| **[Craft](Craft/)** | Alignment-free machine learning for viral subtyping (training and prediction from FASTA/CSV) | [Craft/README.md](Craft/README.md) |
| **[Anniemap](Anniemap/)** | K-mer + FAISS read mapping to compact references, with optional WFA2 alignment | [Anniemap/README.md](Anniemap/README.md) |

Use **Craft** for lineage classification with pretrained or custom models. Use **Anniemap** to map short reads to reference sequences when you need fast k-mer–based placement (and optional SAM output).

## Citations

- van Zyl, D.J., Dunaiski, M., Tegally, H. et al. Alignment-free viral sequence classification at scale. *BMC Genomics* 26, 389 (2025). https://doi.org/10.1186/s12864-025-11554-5
- van Zyl, D.J., Dunaiski, M., Tegally, H. et al. Craft: A Machine Learning Approach to Dengue Subtyping. *bioRxiv* (2025). https://doi.org/10.1101/2025.02.10.637410

## Contact

For Craft-related questions: danielvanzyl@sun.ac.za (see [Craft/README.md](Craft/README.md) for details).
