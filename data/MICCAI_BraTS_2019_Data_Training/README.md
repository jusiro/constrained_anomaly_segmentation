### Experiments on BRATS dataset

To reproduce the experiments carried out in BRATS dataset, you should download the data from the following link: [BRATS](https://drive.google.com/file/d/1NgHMcIcfVGcoAYWd0ABI6AEZCkpFpvJ8/view?usp=sharing). Then, you can adecuate the MRI volumes and produce data slipts using the following code: 

```
python adecuate_BRATS.py --dir_datasets ../data/MICCAI_BraTS_2019_Data_Training/ --dir_out ../data/BRATS_5slices/ --scan flair --nSlices 5
```

For more information on data description and usage requirements please reach out the original authors in the following [LINK](https://www.med.upenn.edu/cbica/brats2020/data.html).
