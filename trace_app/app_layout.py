# Import statements
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
from dash_canvas.utils import array_to_data_url
import dash_mantine_components as dmc
# import dash_extensions.javascript as dj
from .config_ import config

# Update the color constants
COLORS = {
    'background': '#f2f2f2',        # Light gray background
    'panel': '#e6e6e6',             # Slightly darker gray for panels
    'border': '#cccccc',            # Medium gray for borders
    'text': '#333333',              # Dark gray for text
    'card-background': '#e9e9e9',   # Gray for cards
    'white': '#ffffff',             # White
    'tab-background': '#f8f8f8',    # Very light gray for tab background
    'background-light': '#f5f5f5',  # Light background for card headers
    'background-medium': '#eeeeee', # Medium background for card bodies
    'background-dark': '#e0e0e0'    # Dark background for card footers
}

# Style for interactive elements (white background)
INTERACTIVE_STYLE = {
    'backgroundColor': COLORS['white'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '4px'
}

# Card body style
CARD_BODY_STYLE = {
    'backgroundColor': COLORS['background-medium'],
    'padding': '15px',
    'borderRadius': '4px'
}

# Tab styles
TAB_STYLE = {
    'backgroundColor': COLORS['background-medium'],
    'borderBottom': f'1px solid {COLORS["border"]}',
    'padding': '0.5rem 1rem',
    'borderRadius': '4px 4px 0 0'
}

TAB_SELECTED_STYLE = {
    'backgroundColor': COLORS['background-light'],
    'borderBottom': 'none',
    'padding': '0.5rem 1rem',
    'borderRadius': '4px 4px 0 0'
}

# Button style
BUTTON_STYLE = {
    'backgroundColor': COLORS['white'],
    'color': COLORS['text'],
    'border': f'1px solid {COLORS["border"]}',
    'margin': '5px',
    'hover': {
        'backgroundColor': '#f8f9fa'
    }
}

# Card style
CARD_STYLE = {
    'backgroundColor': COLORS['card-background'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '4px',
    'marginBottom': '1rem',
    'padding': '15px'
}

# Control element style for dropdowns, inputs, etc.
CONTROL_STYLE = {
    'backgroundColor': COLORS['white'],
    'borderRadius': '4px',
    'border': f'1px solid {COLORS["border"]}',
    'padding': '6px 12px'
}

# Card styles
CARD_HEADER_STYLE = {
    'backgroundColor': COLORS['background-light'],
    'borderBottom': f'1px solid {COLORS["border"]}',
    'padding': '0.75rem 1.25rem'
}

CARD_BODY_STYLE = {
    'backgroundColor': COLORS['background-medium'],
    'padding': '1.25rem'
}

CARD_FOOTER_STYLE = {
    'backgroundColor': COLORS['background-dark'],
    'borderTop': f'1px solid {COLORS["border"]}',
    'padding': '0.75rem 1.25rem'
}

# Update the style constants
GRAPH_STYLE = {
    'width': '100%',  # Take full width of container
    'height': 'auto',  # Adjust height automatically
    'min-height': '300px',
    'position': 'relative'
}

CONTAINER_STYLE = {
    'background-color': COLORS['background'],
    'padding': '10px'
}

# Add responsive styles
RESPONSIVE_STYLES = {
    '@media (max-width: 768px)': {
        'GRAPH_STYLE': {
            'min-height': '250px',
            'max-height': '400px'
        }
    },
    '@media (max-width: 576px)': {
        'GRAPH_STYLE': {
            'min-height': '200px',
            'max-height': '300px'
        }
    }
}

COREGISTER_CONTAINER_STYLE = {
    'width': '100%',
    'padding': '10px',
    'margin': '0px',
    'display': 'flex',
    'flexDirection': 'column',
    'gap': '10px'
}

# Update the graph style for annotation area
ANNOTATION_GRAPH_STYLE = {
    'width': '100%',
    'height': '35vh',  # Fixed viewport height
    'max-height': 'calc(100vh - 250px)',  # Leave room for controls
    'min-height': '400px',
    'position': 'relative',
    'margin-bottom': '20px'  # Add space before footer
}

DROPDOWN_STYLE = {
    'width': '100%',
    'background-color': 'white'
}

# Update table styles
TABLE_STYLE = {
    'backgroundColor': COLORS['white'],
    'border': f'1px solid {COLORS["border"]}',
    'borderRadius': '4px'
}


# Card components
upload_data_card = dbc.Card(
    children=[
        dbc.CardHeader(html.H2("Upload data area"), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Row([html.Div(id='display_upload_folder_path', children=[])], justify='start'),
            dbc.Row([html.Div("Make a new folder: ", style={'font-weight': 'bold'})], justify='start'),
            dbc.Row([
                dbc.Input(id="new_folder_input", placeholder="Input new folder name", type="text", style={'width': '250px'}),
                dbc.Button("Make new folder", id="make_new_folder_button", size="sm", n_clicks=0, style={'width': '150px'}, className="me-1")
            ], justify='start'),
            dbc.Row([
                html.Div("Data folder you want upload/save to: ", style={'font-weight': 'bold'}),
                dcc.Dropdown(id='upload_data_folder_dropdown', options=config.temp_folder_list, 
                             value='' if len(config.temp_folder_list) == 0 else config.temp_folder_list[0], style={'width': '200px'})
            ], justify='start'),
            dbc.Row([
                html.Div("Upload data: ", style={'font-weight': 'bold'}),
                html.Div(dcc.Upload(id='upload_data_files', children=html.Div(['Drag and Drop or ', html.A('Select File')]),
                                    style={'width': '260px', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                                           'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                                    multiple=False), hidden=True),
                dash_table.DataTable(id='upload_files_table', columns=[{"name": i, "id": i} for i in config.cols],
                                     data=config.upload_files_df_data, row_selectable='multi')
            ], justify='start'),
            dbc.Row([
                dbc.Col(dbc.Button("Upload Whole Slide Image Data", id="upload_button", size="sm", n_clicks=0, style={'width': '200px'}, className="me-1")),
                dbc.Col(dbc.Button("Update File List", id="update_button", size="sm", n_clicks=0, style={'width': '200px'}, className="me-1"))
            ])
        ], style=CARD_BODY_STYLE),
        dbc.CardFooter([
            dbc.Row([
                html.Div([
                html.Div("Upload multiple metals xlsx files and concat to pkl file: ", style={'font-weight': 'bold'}),
                dcc.Upload(id='upload_multiple_xlsx_files', children=html.Div(['Drag and Drop or ', html.A('Select metals xlsx Files')]),
                           style={'width': '320px', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                                  'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'},
                           multiple=True),
                ], hidden=True),
                html.Div("Select samples containing elemental images to upload: ", style={'font-weight': 'bold'}),
                dash_table.DataTable(id='upload_elemental_image_table', columns=[{"name": i, "id": i} for i in config.cols_xlsx],
                                     data=config.upload_files_df_data_xlsx, row_selectable='single'),
                dbc.Button("Upload Elemental Image Data", id="upload_elemental_image_button", size="sm", n_clicks=0, style={'width': '200px'}, className="me-1"),

                

                
            ], justify='start')
        ], style=CARD_FOOTER_STYLE)
    ],
    style=CARD_STYLE
)

data_display_table_card = dbc.Card(
    children=[
        dbc.CardHeader(html.H2("Display data table"), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Row([html.Div(id='display_data_table_container', children=[config.display_files_table])], justify='start')
        ], style=CARD_BODY_STYLE),
        dbc.CardFooter([
            dbc.Row([dbc.Button("Load Selected Data", id="load_data_button", size="sm", n_clicks=0, style={'width': '200px'}, className="me-1")], justify='start'),
            dbc.Row([html.Div(id='output_selected_file', children=[])], justify='start')
        ], style=CARD_FOOTER_STYLE)
    ],
    style=CARD_STYLE
)

image_annotation_card = dbc.Card([
    dbc.CardHeader(html.H2("Annotation area"), style=CARD_HEADER_STYLE),
    dbc.CardBody([
        dcc.Graph(
            id="annotate_metal_image",
            figure=config.blank_figure,
            config={
                "modeBarButtonsToAdd": ["drawrect", "eraseshape", "drawclosedpath"],
                'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d'],
                'responsive': True
            },
            style=ANNOTATION_GRAPH_STYLE
        )
    ], style={**CARD_BODY_STYLE, 'overflow': 'hidden', 'height': '20%'}),
    dbc.CardFooter([
        html.Div([
            html.Div("To annotate the above image", style={'marginBottom': '10px'}),
            html.Div("Choose type:", style={'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Div("Color Map", style={'font-weight': 'bold', 'marginTop': '10px'}),
                        dcc.Dropdown(
                            id='annotation_colormap',
                            options=[{'label': cmap, 'value': cmap} for cmap in plt.colormaps()],
                            value='jet'
                        )
                    ]),
                    width=4
                ),
                dbc.Col(
                    html.Div([
                        html.Div("Metals", style={'font-weight': 'bold', 'marginTop': '10px'}),
                        dcc.Dropdown(
                            id='metal_dropdown_annotation',
                            options=['All'],
                            value='All',
                            searchable=True,
                            style=DROPDOWN_STYLE
                        )
                    ]),
                    width=8
                )
            ], justify="start"),
            dbc.Row([
                html.Div("Display range for concentrations:", style={'font-weight': 'bold'}),
                html.Div(
                    "Adjust percentiles of log(1+concentration) used for display - narrows or widens visible range",
                    style={'font-style': 'italic', 'fontSize': '0.9em', 'marginBottom': '5px'}
                ),
                dcc.RangeSlider(
                    id="vmin_vmax_input",
                    min=0, max=100,
                    value=[0, 100],
                    marks={
                        0: {'label': 'Min', 'style': {'color': '#f50'}},
                        100: {'label': 'Max', 'style': {'color': '#f50'}}
                    },
                    pushable=1,
                    included=False,
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], justify="start"),
            dbc.Row([
                html.Div([  # Wrap threshold controls in hidden div
                    html.Div("Threshold: ", style={'font-weight': 'bold'}),
                    dcc.Slider(
                        id='annotation_threshold_slider',
                        min=0, max=10,
                        step=0.1,
                        value=4,
                        marks={0: '0', 5: '5', 10: '10'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(
                        id={'type': 'threshold-value', 'index': 'annotation'},
                        children='Log threshold: 4.0'
                    )
                ], hidden=True),  # Hide the entire div
            ], justify="start"),
            dcc.Markdown("To annotate the above image\n\n**Choose type**:"),
            dbc.Row([
                dbc.Input(id="type_input", placeholder="Add new type", type="text", style={'width': '170px'}),
                dbc.Button("Add Type", id="add_type_button", size="sm", n_clicks=0, style={'width': '100px'}, className="me-1"),
                dcc.Dropdown(id='type_dropdown_annotation', options=['immune', 'tumor'], value='immune', style={'width': '300px'})
            ]),
            
        ], style={'paddingTop': '20px'})  # Add padding to separate from graph
    ], style=CARD_FOOTER_STYLE)
], style=CARD_STYLE)

# Table and Boxplot components
annotation_result_table = dash_table.DataTable(
    data=config.result_df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in config.result_df.columns],
    fixed_rows={'headers': True},
    style_table={'height': 400, 'overflowY': 'auto', **TABLE_STYLE},
    style_header={
        'backgroundColor': COLORS['white'],
        'fontWeight': 'bold'
    },
    style_data={
        'backgroundColor': COLORS['white']
    }
)

table_result_content = dbc.Card(
    children=[
        dbc.CardHeader(html.H2("Table"), style=CARD_HEADER_STYLE),
        dbc.CardBody([dbc.Row([html.Div(id='table_container', children=annotation_result_table)])], style=CARD_BODY_STYLE),
        dbc.CardFooter([
            dbc.Button("Download Table", id="export_data", size="sm", n_clicks=0, className="me-1"),
            dcc.Download(id="download_dataframe_csv"),
            dcc.Dropdown(
                id='selected_elements',
                options=['All'],
                value=['All'],
                multi=True,
                placeholder="Select elements",
                style={'width': '300px'}
            )
        ], style=CARD_FOOTER_STYLE)
    ],
    style=CARD_STYLE
)

boxplot_result_content = dbc.Card(
    children=[
        dbc.CardHeader(html.H2("Boxplot"), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Row([
                dcc.Graph(
                    id="result_boxplot",
                    config={'responsive': True},
                    style=GRAPH_STYLE
                ),
                dmc.Switch(id="boxplot_metal_type", label="Metals/Annotations", checked=False),
                dmc.Switch(id="boxplot_log_scale", label="Log Scale Boxplot", checked=True),
                dcc.Dropdown(id='boxplot_dropdown', options=['All'], value=['All'], multi=True, style={'width': '300px'})
            ])
        ], style=CONTAINER_STYLE),
        dbc.CardFooter([], style=CARD_FOOTER_STYLE)
    ],
    style=CARD_STYLE
)

table_boxplot_tabs = dbc.Tabs([
    dbc.Tab(table_result_content, label="Table"),
    dbc.Tab(boxplot_result_content, label="Boxplot")
])

# Annotation wsi Card
default_type_color_match_list = []
default_color_to_type_dict = {'#2E91E5': 'immune', '#E15F99': 'tumor'}
for one_color in list(default_color_to_type_dict.keys()):
    new_line = html.P([
        str(default_color_to_type_dict[one_color]) + ': ',
        html.Span("", style={"background-color": one_color, "width": "15px", "height": "15px", "display": "inline-block", "vertical-align": "middle"}),
        f" {one_color}."
    ], style={"display": "flex", "align-items": "center"})
    default_type_color_match_list.append(new_line)

annotation_wsi_card = dbc.Card(
    children=[
        dbc.CardHeader(html.H2("WSI"), style=CARD_HEADER_STYLE),
        dbc.CardBody([
            dbc.Row([
                dcc.Graph(
                    id="blank_wsi_image",
                    figure=config.blank_figure_wsi,
                    config={'responsive': True},
                    style={
                        **GRAPH_STYLE,
                        'height': '200px',
                        'width': '100%',
                        'max-width': '100%',
                        'overflow': 'hidden'
                    }
                )
            ])
        ], style={**CONTAINER_STYLE, 'height': '35%', 'overflow': 'hidden'}),
        dbc.CardFooter([
            dbc.Row([
                html.Div(id="type_color_match_list", children=default_type_color_match_list),
                html.Div("Upload QuPath/ASAP XML or GeoJSON annotation:", style={'font-weight': 'bold', 'margin-top': '10px'}),
                dcc.Upload(id='upload_annotation', children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                           style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                                  'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'},
                           multiple=False)
            ], justify="center")
        ], style={**CARD_FOOTER_STYLE, 'overflow': 'hidden'})
    ],
    style={**CARD_STYLE, 'overflow': 'hidden'}
)

# Layout components
# image_annotation_table = dbc.Row([
#     dbc.Col(annotation_wsi_card, xs=12, sm=12, md=4, lg=4),
#     dbc.Col(image_annotation_card, xs=12, sm=12, md=5, lg=5),
#     dbc.Col(table_boxplot_tabs, xs=12, sm=12, md=3, lg=3)
# ], justify="start", className="g-0")

tab_Measure_content = dbc.Card([
    dbc.CardHeader(html.H2("Measure"), style=CARD_HEADER_STYLE),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(annotation_wsi_card, width=4),
            dbc.Col(image_annotation_card, width=4),
            dbc.Col([
                table_boxplot_tabs,
                dcc.Markdown("**Click to Calculate Results**:", style={'textAlign': 'center'}),
                dbc.Row([
                    dbc.Col([
                        dbc.Button('Calculate Result', id='update_table_button', n_clicks=0, className="me-1", style={'width': '150px'}),
                        dbc.Button("Export", id="export_data_additional", n_clicks=0, className="me-1", style={'width': '150px'}) # size="sm", 
                    ], width="auto", className="d-flex justify-content-center gap-2")
                ], justify="center")
            ], width=4)
        ], className="g-0", style={'minHeight': '80vh'})  # Ensure row has minimum height
    ], style=CARD_BODY_STYLE),
    dbc.CardFooter([], style=CARD_FOOTER_STYLE)
])
all_coregister_content = dbc.Container([
    # Images row
    dbc.Row([
        # Left column with wsi image
        dbc.Col(
            dcc.Graph(
                id='graph_wsi_co',
                figure=config.fig_wsi_co,
                config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False, 'doubleClick': 'reset'},
                style={**GRAPH_STYLE, 'height': '35vh'}
            ),
            width=6
        ),
        # Right column with metal image
        dbc.Col(
            dcc.Graph(
                id='graph_metal_co',
                figure=config.fig_metal_co,
                config={'displayModeBar': False, 'responsive': True, 'scrollZoom': False, 'doubleClick': 'reset'},
                style={**GRAPH_STYLE, 'height': '35vh'}
            ),
            width=6
        )
    ], className="g-0 mb-3"),

    # Controls and table row
    dbc.Row([
        # Left column with co-registration coordinate table
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Co-registration Coordinates", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    html.Div(id='table_container_co', children=config.xy_coords_table)
                ], style=CARD_BODY_STYLE)
            ]),
            width=6
        ),
        # Right column with metal controls
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Metal Controls", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Div("Color Map", style={'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id='metal_colormap_co',
                                    options=[cmap for cmap in plt.colormaps()],
                                    value='jet',
                                    searchable=True
                                )
                            ]),
                            width=4
                        ),
                        dbc.Col(
                            html.Div([
                                html.Div("Metals", style={'font-weight': 'bold'}),
                                dcc.Dropdown(
                                    id='metal_dropdown_co',
                                    options=['All'],
                                    value=['All'],
                                    multi=True
                                )
                            ]),
                            width=8
                        )
                    ]),
                    html.Div("Display range for concentrations:", style={'font-weight': 'bold', 'marginTop': '10px'}),
                    html.Div(
                        "Adjust percentiles of log(1+concentration) used for display - narrows or widens visible range",
                        style={'font-style': 'italic', 'fontSize': '0.9em', 'marginBottom': '5px'}
                    ),
                    dcc.RangeSlider(
                        id="vmin_vmax_input_co",
                        min=0, max=100,
                        value=[0, 100],
                        marks={
                            0: {'label': 'Min', 'style': {'color': '#f50'}},
                            100: {'label': 'Max', 'style': {'color': '#f50'}}
                        },
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div([  # Wrap threshold controls in hidden div
                        html.Div("Visualization threshold:", style={'font-weight': 'bold', 'marginTop': '10px'}),
                        html.Div(
                            "Set threshold for log(1+concentration) - lower values appear white",
                            style={'font-style': 'italic', 'fontSize': '0.9em', 'marginBottom': '5px'}
                        ),
                        dcc.Slider(
                            id='co_threshold_slider',
                            min=0, max=10,
                            step=0.1,
                            value=4,
                            marks={0: '0', 5: '5', 10: '10'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(
                            id={'type': 'threshold-value', 'index': 'co'},
                            children='Log threshold: 4.0'
                        )
                    ], hidden=True),  # Hide the entire div
                ], style=CARD_BODY_STYLE)
            ]),
            width=6
        )
    ], className="g-0 mb-3"),

    # Start co-register button
    dbc.Row([
        dbc.Col(
            dbc.Button('Start Coregister', id="start_co_button", style={**BUTTON_STYLE, 'width': '160px'}, n_clicks=0),
            width='auto'
        )
    ], justify='center', className="mb-3"),

    # View co-registered image
    dbc.Row([
        dbc.Col(html.Div(id='two_image_container'), width=12)
    ], justify='center'),

    dbc.Row([
        dbc.Col(
            dcc.Graph(id='check_border_image', figure=config.white_fig_co, config={'displayModeBar': False}),
            width=12
        )
    ], justify='center')
], fluid=True)

tab_Coregister_content = dbc.Card(dbc.CardBody([all_coregister_content]), className="mt-3")

full_upload_data_card = dbc.Row([
    dbc.Col(upload_data_card, width=6),
    dbc.Col(data_display_table_card, md=6)
], justify="evenly")

tab_Data_content = dbc.Card(dbc.CardBody([full_upload_data_card]), className="mt-3")
# Preprocess tab layout
all_preprocess_content = dbc.Container([
    # Images rows
    dbc.Row([
        # Left image
        dbc.Col(
            html.Div([
                dcc.Graph(
                    id='metal_pre',
                    figure=config.blank_figure,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '35vh', 'width': '100%'}
                )
            ]),
            width=6
        ),
        # Right image
        dbc.Col(
            html.Div([
                dcc.Graph(
                    id='metal_pre_mask',
                    figure=config.blank_figure,
                    config={'displayModeBar': False, 'responsive': True},
                    style={'height': '35vh', 'width': '100%'}
                )
            ]),
            width=6
        )
    ], className="mb-3", style={'height': '35vh'}),

    # Controls row
    dbc.Row([
        # Left column controls
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Image Controls", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    # Colormap and metals dropdowns
                    dbc.Row([
                        dbc.Col(
                            html.Div([
                                html.Div("Color Map", style={'font-weight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='metal_colormap_pre',
                                    options=[cmap for cmap in plt.colormaps()],
                                    value='jet',
                                    searchable=True
                                )
                            ]),
                            width=4
                        ),
                        dbc.Col(
                            html.Div([
                                html.Div("Metals", style={'font-weight': 'bold', 'marginBottom': '5px'}),
                                dcc.Dropdown(
                                    id='metal_dropdown_pre',
                                    options=['All'],
                                    value=['All'],
                                    multi=True
                                )
                            ]),
                            width=8
                        )
                    ]),
                    
                    # Intensity range slider
                    html.Div("Display range for concentrations:", style={'font-weight': 'bold', 'marginTop': '15px'}),
                    html.Div(
                        "Adjust percentiles of log(1+concentration) used for display - narrows or widens visible range",
                        style={'font-style': 'italic', 'fontSize': '0.9em', 'marginBottom': '5px'}
                    ),
                    dcc.RangeSlider(
                        id="vmin_vmax_input_pre",
                        min=0, max=100,
                        value=[0, 100],
                        marks={
                            0: {'label': 'Min', 'style': {'color': '#f50'}},
                            100: {'label': 'Max', 'style': {'color': '#f50'}}
                        },
                        pushable=1,
                        included=False,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    
                    # Threshold slider
                    html.Div([  # Wrap threshold controls in hidden div
                        html.Div("Visualization threshold:", style={'font-weight': 'bold', 'marginTop': '15px'}),
                        html.Div(
                            "Set threshold for log(1+concentration) - lower values appear white",
                            style={'font-style': 'italic', 'fontSize': '0.9em', 'marginBottom': '5px'}
                        ),
                        dcc.Slider(
                            id='pre_threshold_slider',
                            min=0, max=10,
                            step=0.1,
                            value=4,
                            marks={0: '0', 5: '5', 10: '10'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(
                            id={'type': 'threshold-value', 'index': 'pre'},
                            children='Log threshold: 4.0'
                        )
                    ], hidden=True),  # Hide the entire div
                    # html.Div(
                    #     id={'type': 'threshold-value', 'index': 'pre'},
                    #     children='Threshold: 90'
                    # )
                ], style=CARD_BODY_STYLE),
                dbc.CardFooter([], style=CARD_FOOTER_STYLE)
            ]),
            width=6
        ),
        
        # Right column controls
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Mask Generation", style=CARD_HEADER_STYLE),
                dbc.CardBody([
                    # Store for calculated threshold value
                    dcc.Store(id='calculated_threshold', data=None),
                    
                    html.Div([
                        html.Div(
                            "Calculated threshold value for tissue mask generation:",
                            style={'font-weight': 'bold', 'marginBottom': '10px'}
                        ),
                        html.Div(
                            id='threshold_display',
                            style={'font-style': 'italic', 'marginBottom': '15px'}
                        ),
                    ]),
                    html.Div([
                        html.Div(
                            "Select log-scale threshold: tissue will include areas where log(1+concentration) exceeds this value",
                            style={'font-weight': 'bold', 'marginBottom': '10px'}
                        ),
                        dcc.Slider(
                            id="pre_threshold_input", 
                            min=0, max=10,
                            step=0.1,
                            value=4,
                            marks={0: '0', 5: '5', 10: '10'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], hidden=True),
                    dbc.Button(
                        'Generate Mask',
                        id="start_generate_mask_button",
                        style={'width': '200px', 'marginTop': '15px'},
                        n_clicks=0
                    )
                ], style=CARD_BODY_STYLE),
                dbc.CardFooter([], style=CARD_FOOTER_STYLE)
            ]),
            width=6
        )
    ])
], fluid=True)

tab_Preprocess_content = dbc.Card(dbc.CardBody([all_preprocess_content]), className="mt-3")

# all_tabs = dbc.Tabs([
#     dbc.Tab(tab_Data_content, label="Data"),
#     dbc.Tab(tab_Preprocess_content, label="Preprocess"),
#     dbc.Tab(tab_Coregister_content, label="Co-Register"),
#     dbc.Tab(tab_Measure_content, label="Measure")
# ])

# Add CSS to ensure proper scaling
app_style = {
    'width': '100%',
    'max-width': '100vw',
    'margin': '0 auto',
    'padding': '20px',
    'box-sizing': 'border-box',
    'overflow-x': 'hidden'  # Prevents horizontal scrolling
}

# Update the main layout
all_tabs = html.Div([
    dcc.Markdown(children=config.markdown_text_title),
    dbc.Tabs([
        dbc.Tab(
            tab_Data_content,
            label="Data",
            tab_style=TAB_STYLE,
            active_tab_style=TAB_SELECTED_STYLE
        ),
        dbc.Tab(
            tab_Preprocess_content,
            label="Preprocess",
            tab_style=TAB_STYLE,
            active_tab_style=TAB_SELECTED_STYLE
        ),
        dbc.Tab(
            tab_Coregister_content,
            label="Co-Register",
            tab_style=TAB_STYLE,
            active_tab_style=TAB_SELECTED_STYLE
        ),
        dbc.Tab(
            tab_Measure_content,
            label="Measure",
            tab_style=TAB_STYLE,
            active_tab_style=TAB_SELECTED_STYLE
        )
    ], id="main-tabs", style={'backgroundColor': COLORS['tab-background']})
], style={
    **app_style,
    'backgroundColor': COLORS['tab-background'],
    'minHeight': '100vh',
    'padding': '20px'
})
