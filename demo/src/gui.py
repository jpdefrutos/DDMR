import os

import gradio as gr
import numpy as np

from .compute import run_model
from .utils import load_ct_to_numpy


class WebUI:
    def __init__(
        self,
        model_name: str = None,
        cwd: str = "/home/user/app/",
        share: int = 1,
    ):
        # global states
        self.fixed_images = []
        self.moving_images = []
        self.pred_images = []

        # @TODO: This should be dynamically set based on chosen volume size
        self.nb_slider_items = 150

        self.model_name = model_name
        self.cwd = cwd
        self.share = share

        self.class_names = {
            "Brain": "B",
            "Liver": "L"
        }

        # define widgets not to be rendered immediantly, but later on
        self.slider = gr.Slider(
            1,
            self.nb_slider_items,
            value=1,
            step=1,
            label="Which 2D slice to show",
        )

        self.run_btn = gr.Button("Run analysis").style(
            full_width=False, size="lg"
        )

    def set_class_name(self, value):
        print("Changed task to:", value)
        self.class_name = value

    def upload_file(self, files):
        return [f.name for f in files]

    def process(self, mesh_file_names):
        fixed_image_path = mesh_file_names[0].name
        moving_image_path = mesh_file_names[1].name
        output_path = "./"

        run_model(fixed_image_path, moving_image_path, output_path, self.class_names[self.class_name])

        self.fixed_images = load_ct_to_numpy(fixed_image_path)
        self.moving_images = load_ct_to_numpy(moving_image_path)
        self.pred_images = np.ones_like(self.moving_images)
        return self.pred_images

    def get_fixed_image(self, k):
        k = int(k) - 1
        out = [gr.Image.update(visible=False)] * self.nb_slider_items
        out[k] = gr.Image.update(
            self.fixed_images[k],
            visible=True,
        )
        return out
    
    def get_moving_image(self, k):
        k = int(k) - 1
        out = [gr.Image.update(visible=False)] * self.nb_slider_items
        out[k] = gr.Image.update(
            self.moving_images[k],
            visible=True,
        )
        return out
    
    def get_pred_image(self, k):
        k = int(k) - 1
        out = [gr.Image.update(visible=False)] * self.nb_slider_items
        out[k] = gr.Image.update(
            self.pred_images[k],
            visible=True,
        )
        return out

    def run(self):
        css = """
        #model-2d-fixed {
        height: 512px;
        margin: auto;
        }
        #model-2d-moving {
        height: 512px;
        margin: auto;
        }
        #model-2d-pred {
        height: 512px;
        margin: auto;
        }
        #upload {
        height: 120px;
        }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                file_output = gr.File(file_count="multiple", elem_id="upload")
                file_output.upload(self.upload_file, file_output, file_output)

                model_selector = gr.Dropdown(
                    list(self.class_names.keys()),
                    label="Task",
                    info="Which task to perform image-to-registration on",
                    multiselect=False,
                    size="sm",
                )
                model_selector.input(
                    fn=lambda x: self.set_class_name(x),
                    inputs=model_selector,
                    outputs=None,
                )

                self.run_btn.render()

            """
            with gr.Row():
                gr.Examples(
                    examples=[
                        os.path.join(self.cwd, "ixi_image.nii.gz"),
                        os.path.join(self.cwd, "ixi_image2.nii.gz"),
                    ],
                    inputs=file_output,
                    outputs=file_output,
                    fn=self.upload_file,
                    cache_examples=True,
                )
            """

            with gr.Row():
                with gr.Box():
                    with gr.Column():

                        with gr.Row():
                            fixed_images = []
                            for i in range(self.nb_slider_items):
                                visibility = True if i == 1 else False
                                t = gr.Image(
                                    visible=visibility, elem_id="model-2d-fixed"
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                fixed_images.append(t)
                            
                            moving_images = []
                            for i in range(self.nb_slider_items):
                                visibility = True if i == 1 else False
                                t = gr.Image(
                                    visible=visibility, elem_id="model-2d-moving"
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                moving_images.append(t)
                            
                            pred_images = []
                            for i in range(self.nb_slider_items):
                                visibility = True if i == 1 else False
                                t = gr.Image(
                                    visible=visibility, elem_id="model-2d-pred"
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                pred_images.append(t)
                            
                            self.run_btn.click(
                                fn=lambda x: self.process(x),
                                inputs=file_output,
                                outputs=t,
                            )

                        self.slider.input(
                            self.get_fixed_image, self.slider, fixed_images
                        )
                        self.slider.input(
                            self.get_moving_image, self.slider, moving_images
                        )
                        self.slider.input(
                            self.get_pred_image, self.slider, pred_images
                        )

                        self.slider.render()

        # sharing app publicly -> share=True:
        # https://gradio.app/sharing-your-app/
        # inference times > 60 seconds -> need queue():
        # https://github.com/tloen/alpaca-lora/issues/60#issuecomment-1510006062
        demo.queue().launch(
            server_name="0.0.0.0", server_port=7860, share=self.share
        )
