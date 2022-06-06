import io
import uuid

import cv2
import dash
import time

import pandas as pd
from dash.dependencies import Input, Output, State
from dash import dcc, html
import dash_bootstrap_components as dbc
import base64
import tensorflow as tf
from matplotlib import image

from glob import glob
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from matplotlib import pyplot
import time

from evaluateLargeImageProcess import EvaluateLargeImageProcess
from utils.keras.builder import get_backbone_unet
from utils.image import read_image, show_image, array_to_b64, create_dir, trace_boundingBox

# ref: https://python.plainenglish.io/complete-deployment-of-the-image-processing-app-with-code-snippets-2550831bb086
# ref2: https://colab.research.google.com/github/plotly/dash-sample-apps/blob/master/apps/dash-image-enhancing/ColabDemo.ipynb#scrollTo=7wSRcx41FXA5
from utils.runnable import Main

VERSION_APP = 0.1
WEIGHTS_PATH = 'weights/proves_all_data_he_norm_backbone.h5'
ACCEPT_IMAGE_FORMAT = ['jpg', 'jpeg', 'bmp', 'tiff', 'png']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', dbc.themes.LUMEN]


def image_card(img, header=None):
    if img == 'empty':
        card = dbc.CardBody(html.H2(header), className='align-self-center')
    else:
        card = [
            dbc.CardHeader(header, className='align-self-center',
                           style={"fontSize": "20px", "background-color": "white", 'font-weight': 'bold'}),
            dbc.CardImg(src=img, style={"width": "50%", 'align': 'center'}, className='align-self-center')
        ]

    return dbc.Card(card)


class RunMitosisDeepApp(EvaluateLargeImageProcess):
    df = pd.DataFrame(columns=['id'])
    img_path = ''
    output_info = os.path.join('temp', str(uuid.uuid1()))
    model, preprocess = get_backbone_unet(compile_it=False, weights=WEIGHTS_PATH)

    def __init__(self, df=df, patchify_size=256, overlap_patches=1, img_path=img_path,
                 mask_path=None, preprocess=preprocess, model=model, stain=False, stain_ref_img=None,
                 output_info=output_info):
        super().__init__(df, patchify_size, overlap_patches, img_path, mask_path, preprocess, model, stain,
                         stain_ref_img, output_info)

    def build_structure_of_app(self, app):
        # define structure of app
        header = html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=app.get_asset_url('logo.png'), style={'height': '100px',
                                                                               'width': '15%', 'align': 'left'})),

                    dbc.Col(html.H1(children='MitosisDEEP App',
                                    style={
                                        'text-align': 'center',
                                        'color': 'white',
                                        'fontSize': '60px'
                                    }
                                    )),
                    dbc.Col(html.H5(children='version: {}'.format(VERSION_APP),
                                    style={
                                        'text-align': 'right',
                                        'color': 'black'
                                    })),
                ],
                style={
                    'height': 'auto',
                    'width': 'auto',
                    'text-align': 'left',
                    'background-color': '#3E84A7',
                    'align-items': 'center'
                }
            ),
        )

        app.layout = html.Div([
            header,
            dbc.Col(dcc.Upload([
                'Drag and Drop or ', html.A('Select a Image for Mask Mitosis')
            ], style={
                'fontSize': '25px',
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            }, id='upload-image', multiple=False)),
            # html.Button("Download segmentation", id="download-jpg-button", style={'fontSize': '20px'}),
            # dcc.Download(id="download-jpg"),
            html.Button("Save Image to temp directory: {}".format(self.output_info),
                        id="btn_image", style={'fontSize': '20px'}),
            dcc.Download(id="download-image"),
            html.Br(),
            dbc.Col(html.Div(id='output-image-pred'), align='center')

        ])

    def run(self, options):
        app = dash.Dash(external_stylesheets=external_stylesheets)
        self.build_structure_of_app(app)

        @app.callback(Output('output-image-pred', 'children'),
                      Input('upload-image', 'contents'),
                      State('upload-image', 'filename'),
                      prevent_initial_call=True)
        def prediction(contents, filename):
            if contents is None:
                return
            extension = filename.split('.')[-1]
            if extension not in ACCEPT_IMAGE_FORMAT:
                return image_card('empty',
                                  header='Error upload image. \
                                   Only accept images and with extensions: {}'.format(
                                      [accept for accept in ACCEPT_IMAGE_FORMAT]))
            encoded = np.frombuffer(base64.b64decode(contents.split(',')[1]), np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # necessary for nets
            pred_img, size_x, size_y = self.predict_image(img, stain=False)
            img = cv2.resize(img, (size_x, size_y))
            # todo: predict with model
            if (pred_img > 0).any():
                img = trace_boundingBox(img, pred_img, color=(0, 0, 0), width=2)
            str_img = array_to_b64(img, ext="jpeg")
            return image_card(str_img, header='prediction for filename: {}'.format(filename))

        @app.callback(Output("download-image", "data"),
                      Input("btn_image", "n_clicks"),
                      State('output-image-pred', 'children'),
                      State('upload-image', 'filename'),
                      prevent_initial_call=True
                      )
        def func(n_clicks, content, filename):
            if content is None:
                return
            file, ext = filename.split('.')
            contents = content['props']['children'][1]['props']['src']
            encoded = np.frombuffer(base64.b64decode(contents.split(',')[1]), np.uint8)
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # necessary for nets
            cv2.imwrite(os.path.join(self.output_info, file + '_pred.' + 'jpeg'), img)
            # dcc.send_file(temp_img, file + '_pred.' + ext, type=ext)
            # os.remove(temp_img)
            return

        host = os.getenv('DASH_HOST')
        if host is None:
            host = '127.0.0.1'
        app.run_server(debug=True, use_reloader=False, host=host)


if __name__ == '__main__':
    Main(RunMitosisDeepApp()).run()
