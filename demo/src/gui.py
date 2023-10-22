import gradio as gr

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
        self.nb_slider_items = 128

        self.model_name = model_name
        self.cwd = cwd
        self.share = share

        self.class_names = {
            "Brain": "B",
            "Liver": "L"
        }

        self.fixed_image_path = None
        self.moving_image_path = None
        self.fixed_seg_path = None
        self.moving_seg_path = None

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

    def update_fixed(self, cfile):
        self.fixed_image_path = cfile.name
        return self.fixed_image_path
    
    def update_moving(self, cfile):
        self.moving_image_path = cfile.name
        return self.moving_image_path
    
    def update_fixed_seg(self, cfile):
        self.fixed_seg_path = cfile.name
        return self.fixed_seg_path
    
    def update_moving_seg(self, cfile):
        self.moving_seg_path = cfile.name
        return self.moving_seg_path

    def process(self):
        if (self.fixed_image_path is None) or (self.moving_image_path is None):
            raise ValueError("Please, select both a fixed and moving image before running inference.")

        output_path = self.cwd
        
        run_model(self.fixed_image_path, self.moving_image_path, self.fixed_seg_path, self.moving_seg_path, output_path, self.class_names[self.class_name])

        # reset - to avoid using these segmentations again for new images
        self.fixed_seg_path = None
        self.moving_seg_path = None

        self.fixed_images = load_ct_to_numpy(self.fixed_image_path)
        self.moving_images = load_ct_to_numpy(self.moving_image_path)
        self.pred_images = load_ct_to_numpy(output_path + "pred_image.nii.gz")

        return None

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
        #model-2d {
        height: 512px;
        margin: auto;
        }
        #upload {
        height: 80px;
        }
        """
        with gr.Blocks(css=css) as demo:
            with gr.Row():
                
                with gr.Column():
                    file_fixed = gr.File(file_count="single", elem_id="upload", label="Select Fixed Image", show_label=True)
                    file_fixed.upload(self.update_fixed, file_fixed, file_fixed)

                    file_moving = gr.File(file_count="single", elem_id="upload", label="Select Moving Image", show_label=True)
                    file_moving.upload(self.update_moving, file_moving, file_moving)

                #with gr.Group():
                with gr.Column():
                    file_fixed_seg = gr.File(file_count="single", elem_id="upload", label="Select Fixed Seg Image", show_label=True)
                    file_fixed_seg.upload(self.update_fixed_seg, file_fixed_seg, file_fixed_seg)

                    file_moving_seg = gr.File(file_count="single", elem_id="upload", label="Select Moving Seg Image", show_label=True)
                    file_moving_seg.upload(self.update_moving_seg, file_moving_seg, file_moving_seg)

                with gr.Column():
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
                                    visible=visibility, elem_id="model-2d", label="fixed image", show_label=True,
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                fixed_images.append(t)
                            
                            moving_images = []
                            for i in range(self.nb_slider_items):
                                visibility = True if i == 1 else False
                                t = gr.Image(
                                    visible=visibility, elem_id="model-2d", label="moving image", show_label=True,
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                moving_images.append(t)
                            
                            pred_images = []
                            for i in range(self.nb_slider_items):
                                visibility = True if i == 1 else False
                                t = gr.Image(
                                    visible=visibility, elem_id="model-2d", label="predicted fixed image", show_label=True,
                                ).style(
                                    height=512,
                                    width=512,
                                )
                                pred_images.append(t)
                            
                            self.run_btn.click(
                                fn=self.process,
                                inputs=None,
                                outputs=None,
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
