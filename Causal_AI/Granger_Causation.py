# import numpy as np
# import pandas as pd
# from statsmodels.tsa.stattools import grangercausalitytests

# @staticmethod
# def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
#     maxlag=15
#     df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
#     for c in df.columns:
#         for r in df.index:
#             test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
#             p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
#             if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
#             min_p_value = np.min(p_values)
#             df.loc[r, c] = min_p_value
#     df.columns = [var + '_x' for var in variables]
#     df.index = [var + '_y' for var in variables]
#     return df


import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.stattools import grangercausalitytests
import json
import plotly
from plotly.offline import plot

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    maxlag=15
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1], 4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]

    # Create the Plotly heatmap
    fig = px.imshow(df, 
                    labels=dict(x="Predictor", y="Dependent Variable", color="P-Value"),
                    x=df.columns,
                    y=df.index,
                    color_continuous_scale='Viridis',
                    title="Granger Causation Matrix")

    fig.update_layout(
        title_x=0.5,
        width=500,
        height=500
    )
    
    granger_graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    granger_graph_html = plot(fig, output_type='div', include_plotlyjs=False)

    return df, granger_graph_json, granger_graph_html
