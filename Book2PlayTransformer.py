import torch
import torch.nn as nn
import torch.nn.functional as F

# prompt
"""I want to train a single large language model that can handle the multistep pipeline I have set up. I want to be able to input an entire book or short story and then receive a labeled version of the text with a label for each speaking character and an 'other' label for the narrator in order for a full cast audio recording to be generated from. Currently the pipeline is set up as described below:
1. Full text given as input
2. Named entity recogition pipeline to find all full person names.
3. Next we have a zero shot classification pipeline to separate the entire text into dialog and not dialog.
4. Then another zero shot classification pipeline maps the dialog to the character who the dialog belongs to.
5. Generate a useful labeled file to eventually give to a text-to-speech pipeline to 
6. Generate a full cast audio file.
Can you generate a Transformer model to complete this task"""
class Book2PlayTransformer(nn.Module):