import pandas as pd
import plotly.express as px


def create_boxplot(data_dict, title="Gradient Methods Boxplot"):
    df = pd.DataFrame({
        'Method Name': [key for key in data_dict for _ in data_dict[key]],
        'Count of iterations': [item for sublist in data_dict.values() for item in sublist]
    })
    return px.box(
        df,
        x="Method Name",
        y="Count of iterations",
        title=title
    )
