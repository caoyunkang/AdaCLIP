import pandas as pd
import os

def write2csv(results:dict, total_classes, cur_class,  csv_path):
    keys = list(results.keys())

    if not os.path.exists(csv_path):
        df_all = None
        for class_name in total_classes:
            r = dict()
            for k in keys:
                r[k] = 0.00
            df_temp = pd.DataFrame(r, index=[f'{class_name}'])

            if df_all is None:
                df_all = df_temp
            else:
                df_all = pd.concat([df_all, df_temp], axis=0)

        df_all.to_csv(csv_path, header=True, float_format='%.2f')

    df = pd.read_csv(csv_path, index_col=0)

    for k in keys:
        df.loc[f'{cur_class}', k] = results[k]

    df.to_csv(csv_path, header=True, float_format='%.2f')

