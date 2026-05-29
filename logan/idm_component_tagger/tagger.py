from logan.idm_component_tagger.bracket_extractor import extract_bracket_tokens
from logan.idm_component_tagger.config import DEFAULT_COMPONENTS


def _match_component(tokens, component_definitions):
    """Match tokens against seed keywords using substring similarity."""
    best_label = "Other"
    best_score = 0
    for comp_name, keywords in component_definitions.items():
        score = 0
        for tok in tokens:
            for kw in keywords:
                if kw in tok:
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_label = comp_name
    return best_label


class ComponentTagger:
    def __init__(self):
        self.component_definitions = DEFAULT_COMPONENTS

    def tag(self, df):
        """
        Tag each log line with a component using Drain3 template clusters.

        For each unique Drain3 template (test_ids):
          1. Extract bracket tokens from the representative log line
          2. Match tokens against seed keywords using substring similarity
          3. Broadcast the component label to all log lines in that template
        """
        template_component = {}
        for tid, group in df.groupby("test_ids"):
            rep_log = str(group["text"].iloc[0])
            tokens = extract_bracket_tokens(rep_log)
            if tokens:
                template_component[tid] = _match_component(tokens, self.component_definitions)
            else:
                template_component[tid] = "Other"

        df["component"] = df["test_ids"].map(template_component)
        return df
