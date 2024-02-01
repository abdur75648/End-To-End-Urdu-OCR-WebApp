# End-to-End Urdu OCR

[![UTRNet](https://img.shields.io/badge/UTRNet:%20High--Resolution%20Urdu%20Text%20Recognition-blueviolet?logo=github&style=flat-square)](https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)
[![Website](https://img.shields.io/badge/Website-Visit%20Here-darkgreen?style=flat-square)](https://abdur75648.github.io/UTRNet/)
[![arXiv](https://img.shields.io/badge/arXiv-2306.15782-darkred.svg)](https://arxiv.org/abs/2306.15782)
[![SpringerLink](https://img.shields.io/badge/Springer-Page-darkblue.svg)](https://link.springer.com/chapter/10.1007/978-3-031-41734-4_19)
[![SpringerLink](https://img.shields.io/badge/Springer-PDF-blue.svg)](https://rdcu.be/dkbIF)
[![Demo](https://img.shields.io/badge/Demo-Online-brightgreen.svg)](https://abdur75648-urduocr-utrnet.hf.space)

**End-to-End Urdu OCR: A Demo Web App For UTRNet**
This tool was developed by integrating the UTRNet (https://abdur75648.github.io/UTRNe) with a text detection model ([YoloV8](https://docs.ultralytics.com/) finetuned on [UrduDoc](https://paperswithcode.com/dataset/urdudoc)) for end-to-end Urdu OCR.

The application is deployed on Hugging Face Spaces and is available for a live demo. You can access it *[here](https://abdur75648-urduocr-utrnet.hf.space)*. If you prefer to run it locally, you can clone this repository and follow the instructions provided below.

> **Note:** *This version of the application uses a YoloV8 model for text detection. The original version of UTRNet uses ContourNet for this purpose. However, due to deployment issues, we have opted for YoloV8 in this demo. While YoloV8 is as accurate as ContourNet, it offers the advantages of faster processing and greater efficiency.*

## Installation

Clone this repository and install the dependencies using the following command:
> Facing issues in downloading model checkpoints properly? See [this issue]([url](https://github.com/abdur75648/End-To-End-Urdu-OCR-WebApp/issues/1#issuecomment-1920816798))
```bash
pip install -r requirements.txt
```

## Usage
To install the application, first clone this repository. Then, install the necessary dependencies using the following command:
```bash
pip install -r requirements.txt
```

* To run the application, execute the following command:
```bash
python app.py
```

This command launches a Gradio app, which you can interact with to experience the capabilities of UTRNet.

## Citation
If you use the code/dataset, please cite the following paper:

```BibTeX
@InProceedings{10.1007/978-3-031-41734-4_19,
		author="Rahman, Abdur
		and Ghosh, Arjun
		and Arora, Chetan",
		editor="Fink, Gernot A.
		and Jain, Rajiv
		and Kise, Koichi
		and Zanibbi, Richard",
		title="UTRNet: High-Resolution Urdu Text Recognition in Printed Documents",
		booktitle="Document Analysis and Recognition - ICDAR 2023",
		year="2023",
		publisher="Springer Nature Switzerland",
		address="Cham",
		pages="305--324",
		isbn="978-3-031-41734-4",
		doi="https://doi.org/10.1007/978-3-031-41734-4_19"
}
```

### License
[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/). This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/) for Noncommercial (academic & research) purposes only and must not be used for any other purpose without the author's explicit permission.
