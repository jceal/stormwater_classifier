# Main classifier that combines rule-based logic and classifier models

from dataclasses import dataclass
from pathlib import Path

from src.parsing.description_parser import parse_description
from src.models.models import load_models


# Containers for classifier outputs
@dataclass
class FinalLabels:
    ESC: bool
    WQ: bool
    RR: bool
    NNI: object
    Vv: bool


@dataclass
class IntermediateLabels:
    disturb_20000_sf: bool
    new_imp_5000_sf: bool
    new_imp: bool
    table_2_2_activity: bool
    in_ms4: bool
    pollutants_of_concern: list


class StormwaterClassifier:
    """
    Combines rule-based checks with text-based classifiers.
    """

    _cached_models = None

    def __init__(self, lookup_client, models_dir: Path):
        self.lookup = lookup_client
        if StormwaterClassifier._cached_models is None:
            StormwaterClassifier._cached_models = load_models(models_dir)
        self.models = StormwaterClassifier._cached_models

    # Main entry points
    def classify(self, raw_text):
        parsed = parse_description(raw_text)
        loc = self._loc_features(parsed)

        intermediate, vv_pred = self._compute_intermediates(parsed, loc)
        final = self._compute_final_labels(intermediate, vv_pred)
        return final

    def classify_with_explanation(self, raw_text):
        parsed = parse_description(raw_text)
        loc = self._loc_features(parsed)

        intermediate, vv_pred = self._compute_intermediates(parsed, loc)
        final = self._compute_final_labels(intermediate, vv_pred)
        return final, intermediate

    # Helper methods
    def _loc_features(self, parsed):
        if not parsed.street_address or not parsed.borough:
            return {
                "in_ms4_area": False,
                "pollutants_of_concern": [],
                "lot_area_sf": None,
            }
        return self.lookup.lookup_location_features(parsed)

    def _predict(self, key, text):
        model = self.models.get(key)
        if not model:
            return False
        return model.predict_proba([text])[0, 1] >= 0.5

    def _compute_intermediates(self, parsed, loc):
        # Handle case where the entire site is disturbed
        if parsed.disturbed_area_sf == "FULL_SITE":
            disturbed = loc.get("full_site_disturbed_sf", None)
        else:
            disturbed = parsed.disturbed_area_sf or loc.get("lot_area_sf", None)

        new_imp = parsed.new_impervious_sf or 0

        disturbs_20k = disturbed is not None and disturbed >= 20_000
        creates_5k_new_imp = new_imp >= 5_000
        new_imp_any = new_imp > 0

        # Run binary classifiers
        is_table_22 = self._predict("table_2_2_activity", parsed.text)
        vv_pred = self._predict("new_connection", parsed.text)

        intermediate = IntermediateLabels(
            disturb_20000_sf=disturbs_20k,
            new_imp_5000_sf=creates_5k_new_imp,
            new_imp=new_imp_any,
            table_2_2_activity=is_table_22,
            in_ms4=loc.get("in_ms4_area", False),
            pollutants_of_concern=loc.get("pollutants_of_concern", []),
        )

        return intermediate, vv_pred

    # Final rule-based decisions
    def _compute_final_labels(self, I: IntermediateLabels, vv_pred: bool):

        # ESC applies if disturbance or new impervious area exceeds thresholds
        ESC = I.disturb_20000_sf or I.new_imp_5000_sf

        # RR and WQ apply when impervious area is added outside Table 2.2
        RR = (I.new_imp or I.new_imp_5000_sf) and not I.table_2_2_activity
        WQ = RR  # identical SWDM rule

        # NNI applies only for certain MS4 cases
        if (I.new_imp and I.disturb_20000_sf and not I.table_2_2_activity and I.in_ms4):
            NNI = I.pollutants_of_concern
        else:
            NNI = False

        # Vv comes directly from the classifier model
        Vv = vv_pred

        return FinalLabels(ESC=ESC, WQ=WQ, RR=RR, NNI=NNI, Vv=Vv)