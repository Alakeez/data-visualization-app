import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots

continent_options = [{'label': 'World', 'value': 'World'}]
country_options = []
metric_options = [{'label': 'Broadband', 'value': 'Broadband'},
                  {'label': 'Subscriptions', 'value': 'Subscriptions'},
                  {'label': 'Users', 'value': 'Users'},
                  {'label': 'Share', 'value': 'Share'}]
year_options = [{'label': str(year), 'value': str(year)} for year in range(1998, 2018)]

df = pd.read_csv('dataset-full.csv')

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

relative_features = ['Broadband', 'Subscriptions', 'Share']

titles = {
    'Broadband': 'Fixed broadband subscriptions per 100 people',
    'Subscriptions': 'Mobile cellular subscriptions per 100 people',
    'Share': 'Share of individuals using the internet, %',
    'Users': 'Total number of internet users'
}
ytitles = {'Broadband': '', 'Subscriptions': '', 'Share': '', 'Users': ''}

palette = px.colors.diverging.curl
#Rainbow = px.colors.diverging.RdYlBu

def plot_line_plots(metric, countries):
    Figure = go.Figure()
    for idx, country in enumerate(countries):
        Figure.add_scatter(x=df.loc[df['Country'] == country]['Year'].to_numpy(),
                           y=df.loc[df['Country'] == country][metric].to_numpy(),
                           mode='lines',
                           name=country,
                           # marker=dict(
                           #     color=palette[idx % 12]
                           # ),
                           hovertemplate=f'{country}: %{{y:.2f}}<extra></extra>',
                           )

    Figure.update_layout(
        title=titles[metric],
        xaxis_title="Year",
        yaxis_title=ytitles[metric],
        legend_title='Country',
        coloraxis=dict(colorscale=palette),
    )
    return Figure


def plot_bar_plots(metric):
    decades = [1998, 2008, 2017]

    Figure = make_subplots(rows=1, cols=3, shared_yaxes=True)

    Figure.update_yaxes(type='log')

    datasets = []
    for year in decades:
        ds_year_most = df.loc[(df['Year'] == year) & (df[metric] != 0)].sort_values(by=[metric], ascending=False).head()
        ds_year_least = df.loc[(df['Year'] == year) & (df[metric] != 0)].sort_values(by=[metric],
                                                                                     ascending=False).tail()

        ds_year = pd.merge(ds_year_least, ds_year_most, how='outer').sort_values(by=[metric], ascending=True)

        datasets.append(ds_year)

    # for idx, decade in enumerate(decades):
    #     bars = go.Bar(x=datasets[idx][0]['Country'],
    #                   y=datasets[idx][0][metric],
    #                   marker=dict(
    #                       color=palette,
    #                   ),
    #                   showlegend=False,
    #                   width=0.5)
    #     Figure.add_trace(bars, row=1, col=idx+1)
    #
    #     bars = go.Bar(x=datasets[idx][1]['Country'],
    #                   y=datasets[idx][1][metric],
    #                   marker=dict(
    #                       color=palette,
    #                   ),
    #                   showlegend=False,
    #                   width=0.5)
    #     Figure.add_trace(bars, row=2, col=idx+1)

    if metric == 'Users':
        template = '%{x}: %{y}<extra></extra>'
    elif metric == 'Share':
        template = '%{x}: %{y:.2f}<extra></extra>'
    else:
        template = '%{x}: %{y:.4f}<extra></extra>'

    for idx, decade in enumerate(decades):
        bars = go.Bar(x=datasets[idx]['Country'],
                      y=datasets[idx][metric],
                      marker=dict(
                          color=palette[0],
                      ),
                      showlegend=False,
                      width=0.5,
                      hovertemplate=template,)
        Figure.add_trace(bars, row=1, col=idx + 1)
        #
        # bars = go.Bar(x=datasets[idx][1]['Country'],
        #               y=datasets[idx][1][metric],
        #               marker=dict(
        #                   color=palette,
        #               ),
        #               showlegend=False,
        #               width=0.5)
        # Figure.add_trace(bars, row=1, col=idx+1)

    Figure.update_layout(
        title=titles[metric],
        yaxis_title=ytitles[metric],
        legend_title=titles[metric],
        coloraxis=dict(colorscale=palette)
    )

    return Figure


def plot_sunburst_plots(year, metric):
    sunburst_df = df.loc[df['Year'] == year]

    labels = np.append(sunburst_df['Continent'].unique(), sunburst_df['Country'].values)
    parents = np.append(['' for _ in range(len(sunburst_df['Continent'].unique()))],
                        sunburst_df['Continent'].values)

    sunburst_data = dict(type='sunburst',
                         labels=labels,
                         parents=parents,
                         marker=dict(colors=palette),

                         )

    sunburst_layout = dict(
        title=titles[metric],
        legend_title=titles[metric],
        margin=dict(t=50, l=0, r=0, b=0),


    )

    Figure = go.Figure(data=sunburst_data, layout=sunburst_layout)
    Figure.update_traces(
        hoverinfo='text',
        hovertext=np.append([
            continent for continent in sunburst_df['Continent'].unique()],
            [f'{country}: {np.round(metric, 2)}' for country, metric in
             zip(sunburst_df["Country"], sunburst_df[metric])]
        )
    )
    return Figure


def plot_map(year, metric):
    df_map = df.loc[df["Year"] == year]

    Figure = px.choropleth(
        data_frame=df_map,
        locationmode='country names',
        locations='Country',
        scope='world',
        color=metric or 0,
        hover_data=['Country', metric, 'Code', 'Year'],
        color_continuous_scale=palette,
        #color_continuous_scale=palette,
        labels={titles[metric]: metric},
        template='plotly_white',
    )
    Figure.update_geos(fitbounds="locations", visible=False)
    Figure.update_layout(
        title=titles[metric],
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        geo=dict(
            landcolor='lightgray',
            showland=True,
            showcountries=True,
            countrycolor='gray',
            countrywidth=0.5,
            projection=dict(
                type='natural earth'
            ))
    )

    return Figure


bar_plots = plot_bar_plots('Broadband')
sunburst_plots = plot_sunburst_plots(2017, 'Broadband')
map_plot = plot_map(2017, 'Broadband')

app.layout = html.Div([

    html.Div([

        html.Div(id='head-container', children=[]),
        html.Br(),

        html.Label('The Internet in the World (1998-2017)',style={'textAlign': 'center','fontSize': 40}),
        html.Label(' The creation of the World Wide Web in 1989 by Tim Berbers-Lee revolutionized our history of communication.',style={'textAlign': 'center','fontSize': 20}),
        html.Label('Here we want to look at the global expansion of the internet since then.',style={'textAlign': 'center','fontSize':20}),
    ]),

    html.Br(),
    html.Hr(),

    html.Div([

        html.Div(id='map-container', children=[]),
        html.Br(),

        dcc.Graph(id='world-map', figure={}),

        html.Label('Year'),
        dcc.Slider(
            id='map-year-slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            marks={str(i): str(i) for i in range(df['Year'].min(), df['Year'].max() + 1)},
            value=df['Year'].min(),
            step=1
        ),

        html.Br(),

        html.Label('Metric'),
        dcc.Dropdown(
            id='map-metric-drop',
            options=metric_options,
            value='Broadband',
        ),

    ]),

    html.Br(),
    html.Hr(),

    html.Div([
        html.Label('Continent'),
        dcc.Dropdown(
            id='continent-drop',
            options=continent_options,
            value='Europe',
        ),

        html.Label('Country'),
        dcc.Dropdown(
            id='country-drop',
            options=country_options,
            multi=True,
            value='Portugal',
        ),

        html.Label('Metric'),
        dcc.Dropdown(
            id='line-metric-drop',
            options=metric_options,
            value='Broadband',
        ),

        html.Div(id='line-container', children=[]),
        html.Br(),

        dcc.Graph(
            id='line-graph',
        ),

    ]),

    html.Br(),
    html.Hr(),

    html.Div([

        html.Div([

            html.Div(id='sunburst-container', children=[]),
            html.Br(),

            dcc.Graph(
                id='sunburst-graph',
                figure=sunburst_plots,
            ),

            html.Label('Year'),
            dcc.Slider(
                id='sunburst-year-slider',
                min=df['Year'].min(),
                max=df['Year'].max(),
                marks={str(i): str(i) for i in range(df['Year'].min(), df['Year'].max() + 1)},
                value=df['Year'].min(),
                step=1
            ),

            html.Label('Metric'),
            dcc.Dropdown(
                id='sunburst-metric-drop',
                options=metric_options,
                value='Broadband',
            ),

        ]),

        html.Div([
            html.Div(id='side-container', children=[]),
            html.Br(),

            html.Label('By 1998 only 56 countries had internet users. 20 years later the internet has arrived to all countries but not to everyone.',style={'textAlign': 'left','fontSize': 20}),
        ]),

    ]),

    html.Br(),
    html.Hr(),

    html.Div([

        html.Div(id='decades-container', children=[]),
        html.Br(),

        dcc.Graph(
            id='bar-graph',
            figure=bar_plots,
        ),

        html.Label('Metric'),
        dcc.Dropdown(
            id='bar-metric-drop',
            options=metric_options,
            value='Broadband',
        ),

    ]),

    html.Br(),
    html.Hr(),

    html.Div([
        html.Div(id='footer-container', children=[]),
        html.Br(),

        html.Label('Authors: Maiia Tagunova,20210984/Ramzi Al-Ayass,20210705/Vera Canhoto,20210659.',style={'textAlign':'center','fontSize': 20}),
        html.Label('Sources: https:// https://ourworldindata.org/internet',style={'textAlign': 'center','fontSize': 20})
    ]),
])


@app.callback(
    Output('country-drop', 'options'),
    Input('continent-drop', 'value'),
    State('country-drop', 'value')
)
def update_multi_options(continent, selected_countries):
    if not continent:
        raise PreventUpdate

    if continent == 'World':
        continent_countries = df['Country'].unique()
    else:
        continent_countries = df.loc[df['Continent'] == continent]['Country'].unique()
    countries = set(country
                    for country in continent_countries
                    )
    if selected_countries is not None:
        for selected_country in selected_countries:
            if selected_country in continent_countries:
                countries.add(selected_country)
    options = [{'label': country, 'value': country} for country in countries]
    return options


@app.callback(
    Output(component_id='line-graph', component_property='figure'),
    [Input(component_id='line-metric-drop', component_property='value'),
     Input(component_id='country-drop', component_property='value')
     ]
)
def update_line_plots(metric, options):
    if not isinstance(options, list):
        options = [options]
    return plot_line_plots(metric, options)


@app.callback(
    Output(component_id='bar-graph', component_property='figure'),
    Input(component_id='bar-metric-drop', component_property='value'),
)
def update_bar_plots(metric):
    return plot_bar_plots(metric)


@app.callback(
    Output(component_id='sunburst-graph', component_property='figure'),
    [Input(component_id='sunburst-year-slider', component_property='value'),
     Input(component_id='sunburst-metric-drop', component_property='value'),
     ]
)
def update_sunburst_plots(year, metric):
    return plot_sunburst_plots(int(year), metric)


@app.callback(
    # [Output(component_id='map-container', component_property='children'),
    Output(component_id='world-map', component_property='figure'),
    # ],
    [Input(component_id='map-year-slider', component_property='value'),
     Input(component_id='map-metric-drop', component_property='value')]
)
def update_graph(year, metric):
    # return f"{titles[metric]}, year {year}", \
    return plot_map(year, metric)


if __name__ == '__main__':
    for continent in df['Continent'].unique():
        continent_options.append({'label': continent, 'value': continent})

    for country in df['Country'].unique():
        country_options.append({'label': country, 'value': country})

    app.run_server(debug=True)
