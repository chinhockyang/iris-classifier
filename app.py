# General / Data Science / Machine Learning
import pandas as pd
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Charts
import plotly_express as px
from plotly.figure_factory import create_distplot
import plotly.graph_objects as go

# Dashboard
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import warnings
warnings.filterwarnings("ignore")


# create dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server

# ************************************ DATA PARAMETERS ************************************ #

IRIS_CSV_DATA_FOLDER = "data"
IRIS_CSV_DATA_FILENAME = "iris.csv"

MODEL_PKL_FOLDER = "model"
MODEL_PKL_FILENAME = "best_model.pkl"

# ****************************************************************************************** #

IRIS_DATA_PATH = os.path.join(os.getcwd(), IRIS_CSV_DATA_FOLDER, IRIS_CSV_DATA_FILENAME)
MODEL_PKL_PATH = os.path.join(os.getcwd(), MODEL_PKL_FOLDER, MODEL_PKL_FILENAME)

# Load Model
def load_file(path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data

model = load_file(MODEL_PKL_PATH)

# Load Dataset
df = pd.read_csv(IRIS_DATA_PATH, header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "type"])

# Normalisation
for column in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
    df[f'normalised_{column}'] = (df[column] - df[column].min()) / (df[column].max() - df[column].min()) 

# Define Dictionary to display "Pretty Name" on Dashboard
dct_name_mapper = {
    'Iris-virginica': 'Virginica', 
    'Iris-setosa': 'Setosa',
    'Iris-versicolor': 'Versicolor',
    'Virginica': 'Iris-virginica',
    'Setosa': 'Iris-setosa',
    'Versicolor': 'Iris-versicolor',
}

dct_feature_name_mapper = {
    "sepal_length": "Sepal Length",
    "sepal_width": "Sepal Width",
    "petal_length": "Petal Length",
    "petal_width": "Petal Width",
    "normalised_sepal_length": "Normalised Sepal Length",
    "normalised_sepal_width": "Normalised Sepal Width",
    "normalised_petal_length": "Normalised Petal Length",
    "normalised_petal_width": "Normalised Petal Width"
}

color_scheme = {
    'Setosa': '#4C3277',
    'Versicolor': '#FFB6A2',
    'Virginica': '#FF4674',
}

# ****************************** CALLBACK FUNCTIONS START **************************** #

# ------------ Display Propotion of Iris Types in Data ----------- #
def return_basic_pie_chart():
    fig = px.pie(
        df["type"].value_counts().reset_index().replace(dct_name_mapper).rename(columns={"index": "Type", "type": "Count"}),
        values="Count",
        names="Type",
        color="Type",
        color_discrete_map={
            "Setosa": color_scheme["Setosa"],
            "Versicolor": color_scheme["Versicolor"],
            "Virginica": color_scheme["Virginica"],
        },
        title="Number of Iris of each Type"
    )
    return fig

# ------------ Display Number of Available Data for each Iris Types ----------- #
@app.callback(        
    Output(component_id='data-availability', component_property='figure'),
    [Input(component_id='data-availability-dropdown', component_property='value'),
    ],
)
def return_data_availability(type):
    df_subset = df[df["type"] == type]

    total_record = len(df_subset)
    num_of_missing_sepal_length = len(df_subset[df_subset["sepal_length"].isnull()])
    num_of_missing_sepal_width = len(df_subset[df_subset["sepal_width"].isnull()])
    num_of_missing_petal_length = len(df_subset[df_subset["petal_length"].isnull()])
    num_of_missing_petal_width = len(df_subset[df_subset["petal_width"].isnull()])

    df_availability = pd.DataFrame({
        "Count": [ total_record - num_of_missing_sepal_length, total_record - num_of_missing_sepal_width,
        total_record - num_of_missing_petal_width, total_record - num_of_missing_petal_length,
        num_of_missing_sepal_length, num_of_missing_sepal_width, num_of_missing_petal_width, 
        num_of_missing_petal_length],
        "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"] * 2,
        "Type": ["Available"] * 4 + ["Missing"] * 4
    })

    fig = px.bar(
        df_availability,
        x = "Count",
        y = "Feature",
        color="Type",
        barmode="group",
        text_auto='.1s',
        color_discrete_map={
            "Available": color_scheme["Setosa"],
            "Missing": color_scheme["Versicolor"]
        },
        title = f"Number of Available and Missing Data for {dct_name_mapper[type]} Iris"
    )
    fig.update_traces(marker_line_color = 'black', marker_line_width = 1.1)
    return fig

# ------------ Toggle the Displayed Figure for Distribution Chart (Below) ----------- #
@app.callback(
   Output(component_id='distribution-type-dropdown', component_property='style'),
   [Input(component_id='distribution-radio', component_property='value')]
)
def display_type_dropdown(chart):
    if chart == 'attribute':
        return {'display': 'block'}
    if chart == 'feature':
        return {'display': 'none'}

@app.callback(
   Output(component_id='distribution-feature-dropdown', component_property='style'),
   [Input(component_id='distribution-radio', component_property='value')]
)
def display_feature_dropdown(chart):
    if chart == 'feature':
        return {'display': 'block'}
    if chart == 'attribute':
        return {'display': 'none'}

    
# ------------  Distribution Chart to compare across Features/Types of Iris ----------- #
@app.callback(        
    Output(component_id='distribution-across-features', component_property='figure'),
    [Input(component_id='distribution-radio', component_property='value'),
    Input(component_id='distribution-type-dropdown', component_property='value'),
    Input(component_id='distribution-feature-dropdown', component_property='value')
    ],
)
def return_distribution_across_features(chart, type, feature):
    if chart == "attribute":
        df_subset = df[df["type"] == type]
    
        fig = create_distplot(
            [df_subset["sepal_length"], df_subset["sepal_width"], df_subset["petal_length"], df_subset["petal_width"]], 
            ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"], colors = list(color_scheme.values()) + ["#93932A"],show_hist=False)
        fig.update_layout(title=f"Distribution of Attributes of {dct_name_mapper[type]}")
        return fig
    else:
        df_setosa = df[df["type"] == "Iris-setosa"]
        df_versicolor = df[df["type"] == "Iris-versicolor"]
        df_virginica = df[df["type"] == "Iris-virginica"]

        fig = create_distplot(
            [df_setosa[feature], df_versicolor[feature], df_virginica[feature]], 
            ["Setosa", "Versicolor", "Virginica"], colors = list(color_scheme.values()),show_hist=False)
        fig.update_layout(title=f"Distribution of {dct_feature_name_mapper[feature]} across Features (Types)")
        return fig

def return_distribution_across_types(df, feature):
    df_setosa = df[df["type"] == "Iris-setosa"]
    df_versicolor = df[df["type"] == "Iris-versicolor"]
    df_virginica = df[df["type"] == "Iris-virginica"]

    fig = create_distplot(
        [df_setosa[feature], df_versicolor[feature], df_virginica[feature]], 
        ["Setosa", "Versicolor", "Virginica"], colors = list(color_scheme.values()),show_hist=False)
    fig.update_layout(title=f"Distribution of {dct_feature_name_mapper[feature]} between Iris Types")


# ------------ Create Radar Chart to Compare Across Iris Types ----------- #
def return_overall_radar_chart():
    df_melt_normalised = pd.melt(df, id_vars=["type"], value_vars=["normalised_sepal_length", "normalised_sepal_width", "normalised_petal_length", "normalised_petal_width"])
    df_melt_normalised = df_melt_normalised.groupby(["type", "variable"]).mean().reset_index().replace(dct_feature_name_mapper)
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df_melt_normalised.query("type=='Iris-versicolor'")["value"],
        theta=df_melt_normalised.query("type=='Iris-versicolor'")["variable"],
        fill="toself",
        name="Versicolor"
    ))

    fig.add_trace(go.Scatterpolar(
        r=df_melt_normalised.query("type=='Iris-virginica'")["value"],
        theta=df_melt_normalised.query("type=='Iris-virginica'")["variable"],
        fill="toself",
        name="Virginica"
    ))

    fig.add_trace(go.Scatterpolar(
        r=df_melt_normalised.query("type=='Iris-setosa'")["value"],
        theta=df_melt_normalised.query("type=='Iris-setosa'")["variable"],
        fill="toself",
        name="Setosa"
    ))

    fig.update_layout(
        title="Comparison across Iris Types",
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    )
    return fig


# ------------ Return Top 10 Similar Data and Predicted Iris Type of Input Data ----------- #
@app.callback(        
    Output(component_id='update-table', component_property='children'),
    Input(component_id='submit', component_property='n_clicks'),
    [
    State(component_id='input1', component_property='value'), State(component_id='input2', component_property='value'),
    State(component_id='input3', component_property='value'), State(component_id='input4', component_property='value'),
    ],
)
def return_top_ten_similar_data(n_clicks, input1, input2, input3, input4):
    # No Input were provided
    if (pd.isna(input1) or pd.isna(input2) or pd.isna(input3) or pd.isna(input4)):
        return html.P("")

    new_data = {
        "sepal_length": [input1],
        "sepal_width": [input2],
        "petal_length": [input3],
        "petal_width": [input4],
    }
    
    # Normalise data in existing data table and the new input data (separately)
    old_df = df.copy()
    for column in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        old_df[f'normalised_{column}'] = (old_df[column] - old_df[column].min()) / (old_df[column].max() - old_df[column].min()) 
        new_data[f"normalised_{column}"] = (new_data[column] -  old_df[column].min()) / (old_df[column].max() - old_df[column].min())

    # Add new input data to the first row of Data Table (and reset index so that it has an index of 0)
    df_new = pd.concat([
        pd.DataFrame(new_data),
        old_df
    ])
    df_new.reset_index(drop=True, inplace=True)

    # Generate Probability of Predicted Type using the Logistic Regression Model
    predict_proba_matrix = model.predict_proba(df_new[["normalised_sepal_length", "normalised_sepal_width", "normalised_petal_length", "normalised_petal_width"]])
    df_predict_proba = pd.DataFrame(predict_proba_matrix, columns=["proba_setosa", "proba_versicolor", "proba_virginica"])

    # Identify the Predicted Iris Class of the Input Data
    df_pred = df_predict_proba.iloc[0,:].reset_index()\
                .rename(columns={"index":"Type", 0: "Probability"})\
                .replace({
                    "proba_setosa": "Setosa",
                    "proba_versicolor": "Versicolor",
                    "proba_virginica": "Virginica"
                })
    predicted_type = df_pred.sort_values("Probability", ascending=False).iloc[0,:]["Type"]
    predicted_type_raw = dct_name_mapper[predicted_type]
    
    # Generate Pie Chart to visualise Predicted Probability
    fig = px.pie(df_pred, values="Probability", names="Type", hole=.3, 
                color="Type",
                color_discrete_map={
                        "Setosa": color_scheme["Setosa"],
                        "Versicolor": color_scheme["Versicolor"],
                        "Virginica": color_scheme["Virginica"],
                })
    
    # Data Transformation to compare New Input Data and the Average of the Existing Data of the Predicted Type
    df_melt_original = pd.melt(old_df[old_df["type"] == predicted_type_raw], id_vars=["type"], value_vars=["normalised_sepal_length", "normalised_sepal_width", "normalised_petal_length", "normalised_petal_width"]) \
        .groupby(["type", "variable"]).mean().reset_index().replace(dct_feature_name_mapper)
    df_melt_original["type"] = "Mean"
    df_input_data = df_new.loc[0, ["normalised_sepal_length", "normalised_sepal_width", "normalised_petal_length","normalised_petal_width"]]\
            .reset_index()\
            .rename(columns={"index": "variable", 0: "value"})\
            .replace(dct_feature_name_mapper)
    df_input_data["type"] = "Input"
    df_radar = pd.concat([
        df_melt_original,df_input_data
    ])

    # Generate Radar Chart to visualise Comparison
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=df_radar.query("type=='Input'")["value"],
        theta=df_radar.query("type=='Input'")["variable"],
        fill="toself",
        name="Input Data"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=df_radar.query("type=='Mean'")["value"],
        theta=df_radar.query("type=='Mean'")["variable"],
        fill="toself",
        name="Mean Data"
    ))
    fig_radar.update_layout(
        title=f"Similarity to Existing {predicted_type}",
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
        ))
    )
    
    # Use Cosine Similarity to Compute Similarity Score between Input Data and Existing Data
    df_combined = pd.concat([df_new, df_predict_proba], axis=1)
    similarity_matrix = cosine_similarity(df_combined[["normalised_sepal_length", "normalised_sepal_width", "normalised_petal_length", "normalised_petal_width", "proba_setosa","proba_versicolor","proba_virginica"]])
    index = pd.DataFrame(similarity_matrix[0]).sort_values(by=0, ascending=False).loc[0:,].iloc[1:11,:].index
    df_output = df_combined.loc[index, :]

    # Return top 10 similar data
    df_top_ten = df_output[["sepal_length", "sepal_width", "petal_length", "petal_width","type"]]
    df_top_ten.rename(columns={
        "sepal_length": "Sepal Length",
        "sepal_width": "Sepal Width",
        "petal_length": "Petal Length",
        "petal_width": "Petal Width",
        "type": "Type"
    }, inplace=True)
    df_top_ten.replace(dct_name_mapper, inplace=True)
    
    return html.Div(
        [
            dbc.Row([
                dbc.Col([
                    html.H5(children="Predicted Type:"),
                    html.H3(children=predicted_type),
                    dcc.Graph(
                        figure=fig
                    )
                ]),
                dbc.Col([
                    dcc.Graph(
                        figure=fig_radar
                    )
                ])
            ]),
            html.H5(children='Top 10 Similar Data Points', className="text-center"),
            dbc.Table.from_dataframe(df_top_ten)
        ]
    )

###############################################################


# ************************************ APP START ************************************ #


# app layout
app.layout = html.Div(children=[    
    dbc.Container([
           html.H2(children='Iris Data Report', className="lead display-3 text-center py-3"),
           html.Hr(className="my-4"),
           html.P(children='Chin Hock Yang', className="lead text-center text-xl pb-3"),            
    ], className="bg-light shadow-sm my-3 w-75 mx-auto"),

    dbc.Container([
            html.H2(children='Data Exploration', className="text-center bg-primary text-light py-3"),            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=return_basic_pie_chart()
                    ),
                ], width = 6),
                dbc.Col([
                    html.H6(children='Select Iris Type', className="mt-1"),
                    dcc.Dropdown(
                        id='data-availability-dropdown',
                        options=[
                            {'label': 'Setosa', 'value': 'Iris-setosa'},
                            {'label': 'Versicolor', 'value': 'Iris-versicolor'},
                            {'label': 'Virginica', 'value': 'Iris-virginica'}
                        ],
                        value="Iris-setosa"
                    ),
                    dcc.Graph(
                        id='data-availability',
                        figure={}
                    )
                ], width = 6)
            ], className="bg-white w-40 mx-auto"),
    ], className="bg-white mb-3 pb-3"),

    dbc.Container([
            html.H2(children='Comparison across Iris Types', className="text-center bg-primary text-light py-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=return_overall_radar_chart()
                    )
                ], width = 6),
                dbc.Col([
                        dcc.RadioItems(
                            id='distribution-radio',
                            options=[
                                {'label': html.Span("Feature (Type of Iris)", style={ 'margin-right': '10px', 'margin-left': '2px'}), 'value': 'feature'},
                                {'label': html.Span("Attribute (Characteristics of Iris)", style={ 'margin-right': '10px', 'margin-left': '2px'}), 'value': 'attribute'},
                            ],
                            value="feature"
                        ),
                        html.Div([
                            dcc.Dropdown(
                                id='distribution-type-dropdown',
                                options=[
                                    {'label': 'Setosa', 'value': 'Iris-setosa'},
                                    {'label': 'Versicolor', 'value': 'Iris-versicolor'},
                                    {'label': 'Virginica', 'value': 'Iris-virginica'}
                                ],
                                value="Iris-setosa"
                            )
                            ], style = { 'display': 'block', 'marginTop': 3 }
                        ),
                        html.Div([
                            dcc.Dropdown(
                                id='distribution-feature-dropdown',
                                options=[
                                    {'label': 'Sepal Length', 'value': 'sepal_length'},
                                    {'label': 'Sepal Width', 'value': 'sepal_width'},
                                    {'label': 'Petal Length', 'value': 'petal_length'},
                                    {'label': 'Petal Width', 'value': 'petal_width'}
                                ],
                                value="sepal_length"
                            )
                            ], style = { 'display': 'block' }
                        ),
                        dcc.Graph(        
                            id="distribution-across-features",
                            figure={}
                        )
                ], width = 6)
            ], justify="center", align="center", className="bg-white w-40 h-80 mx-auto"),

            dbc.Container([
                html.H2(children='New Data Prediction', className="text-center bg-primary text-light py-3"),
                html.H6(children='Enter the value of the new Iris Data', className="text-center mt-3"),
                dbc.Row([
                    dbc.Col([
                        html.P("Sepal Length"),
                        dcc.Input(id="input1", type="number", placeholder="", debounce=True)
                    ]),
                    dbc.Col([
                        html.P("Sepal Width"),
                        dcc.Input(id="input2", type="number", placeholder="", debounce=True)
                    ]),
                    dbc.Col([
                        html.P("Petal Length"),
                        dcc.Input(id="input3", type="number", placeholder="", debounce=True)
                    ]),
                    dbc.Col([
                        html.P("Petal Width"),
                        dcc.Input(id="input4", type="number", placeholder="", debounce=True)
                    ]),
                ], className="bg-light py-3")
            ], style= { "display" : "inline"}),
            dbc.Button("Predict", id='submit', n_clicks=0, className="mt-3 d-flex justify-content-center w-50 mx-auto", color="secondary"),
        
            dbc.Container([
                html.Div(id="update-table")
            ], class_name="my-3")

    ], className="bg-white w-40 h-80 mb-4 mx-auto")

])

# ************************************ APP ENDS ************************************ #
if __name__ == '__main__':
    app.run_server(debug=True)
