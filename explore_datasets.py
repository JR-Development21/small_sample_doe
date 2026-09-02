import pandas as pd
from openml import datasets

datalist = datasets.list_datasets(output_format="dataframe")
datalist = datalist[(datalist.NumberOfMissingValues == 0) &
                    (datalist.NumberOfInstances < 100000) &
                    (datalist.MinorityClassSize >= 500) & 
                    (~datalist.MinorityClassSize.isna()) & 
                    (datalist.status == "active") & 
                    (datalist.NumberOfNumericFeatures > 10) & 
                    (datalist.NumberOfSymbolicFeatures < 10) &
                    (datalist.NumberOfFeatures > 100) &
                    (datalist.NumberOfClasses == 2)]

datalist[["did", "name", "NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"]]

X, y, categorical_indicator, attribute_names = datasets.get_dataset(46859).get_data()