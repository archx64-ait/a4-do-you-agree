from django.views.generic import TemplateView
from django.views.generic.edit import FormView
from nlp.forms import NLPForm
from typing import Any
from nlp.utils_scratch import *
import torch


class IndexView(TemplateView):
    template_name = "index.html"


# class SuccessView(TemplateView):
#     template_name = "success.html"

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)

#         result = self.request.GET.get("result")

#         try:
#             # Add the result to the context
#             context["result"] = result

#         except ValueError:
#             context["result"] = [""]

#         return context


class NLPFormView(FormView):

    form_class = NLPForm
    template_name = "nlp.html"
    seq_length = 128
    label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

    model = BERT(
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_ff=d_ff,
        d_k=d_k,
        n_segments=n_segments,
        vocab_size=len(word2id),
        max_len=max_len,
        device=device,
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    classifier_head = nn.Linear(768 * 3, 3).to(device)
    classifier_head.load_state_dict(
        torch.load(CHEAD_PATH, map_location=device)
    )
    classifier_head.eval()

    
    def predict_nli(self, premise, hypothesis):
        inputs_a = custom_tokenizer(premise)
        inputs_b = custom_tokenizer(hypothesis)

        # convert lists to torch tensors and move to device
        input_ids_a = torch.tensor(inputs_a["input_ids"]).to(device)
        attention_a = torch.tensor(inputs_a["attention_mask"]).to(device)
        input_ids_b = torch.tensor(inputs_b["input_ids"]).to(device)
        attention_b = torch.tensor(inputs_b["attention_mask"]).to(device)

        # use the model's helper method to get token embeddings
        u = self.model.get_last_hidden_state(input_ids_a)
        v = self.model.get_last_hidden_state(input_ids_b)

        # get mean pooled sentence embeddings
        u_mean_pool = mean_pool(u, attention_a)
        v_mean_pool = mean_pool(v, attention_b)

        # build the |u-v| tensor
        uv_abs = torch.abs(u_mean_pool - v_mean_pool)

        # concatenate u, v, |u-v|
        x = torch.cat([u_mean_pool, v_mean_pool, uv_abs], dim=-1)

        # process concatenated tensor through classifier_head
        logits = self.classifier_head(x)

        #get the predicted label
        pred_label = torch.argmax(logits, dim=1).item()
        pred_class = self.label_map[pred_label]
        return pred_class

    def form_valid(self, form):
        premise = form.cleaned_data["premise"]
        hypothesis = form.cleaned_data["hypothesis"]
        result = self.predict_nli(premise=premise, hypothesis=hypothesis)
        context = self.get_context_data(result=result)
        print(context)
        return self.render_to_response(context)

    def form_invalid(self, form):
        return super().form_invalid(form)

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        # context["results"] = getattr(self, "result", None)
        context["result"] = kwargs.get("result", None)
        return context
