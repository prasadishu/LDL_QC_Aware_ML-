    import numpy as np

class WestgardQC:
    """
    Westgard-inspired QC engine for post-prediction validation
    """

    def __init__(self, mean, sd):
        """
        mean : expected analytical mean (from training / lab validation)
        sd   : analytical standard deviation
        """
        self.mean = mean
        self.sd = sd

    # -------- Individual Westgard Rules -------- #

    def rule_1_2s(self, value):
        """ Warning rule """
        return abs(value - self.mean) > 2 * self.sd

    def rule_1_3s(self, value):
        """ Reject rule """
        return abs(value - self.mean) > 3 * self.sd

    def rule_2_2s(self, history):
        """ Two consecutive results >2SD on same side """
        if len(history) < 2:
            return False

        return (
            history[-1] > self.mean + 2*self.sd and
            history[-2] > self.mean + 2*self.sd
        ) or (
            history[-1] < self.mean - 2*self.sd and
            history[-2] < self.mean - 2*self.sd
        )

    def rule_r_4s(self, history):
        """ Difference between two consecutive results >4SD """
        if len(history) < 2:
            return False

        return abs(history[-1] - history[-2]) > 4 * self.sd

    def rule_4_1s(self, history):
        """ Four consecutive results >1SD on same side """
        if len(history) < 4:
            return False

        last = history[-4:]
        return (
            all(v > self.mean + self.sd for v in last) or
            all(v < self.mean - self.sd for v in last)
        )


# -------- QC Decision Logic -------- #

def apply_qc(predicted_value, prediction_history, qc_engine):
    """
    Applies Westgard rules and returns QC decision

    Returns:
    - qc_status: PASS / FAIL
    - violated_rules: list of violated rules
    """

    violations = []

    if qc_engine.rule_1_3s(predicted_value):
        violations.append("1-3s")

    if qc_engine.rule_2_2s(prediction_history):
        violations.append("2-2s")

    if qc_engine.rule_r_4s(prediction_history):
        violations.append("R-4s")

    if qc_engine.rule_4_1s(prediction_history):
        violations.append("4-1s")

    # 1-2s is warning only
    warning = qc_engine.rule_1_2s(predicted_value)

    if violations:
        return {
            "qc_status": "FAIL",
            "violations": violations,
            "warning": warning
        }

    return {
        "qc_status": "PASS",
        "violations": None,
        "warning": warning
    }


# -------- Conservative Adjustment -------- #

def conservative_adjustment(value, mean):
    """
    Pulls prediction towards expected mean
    Used ONLY when QC fails
    """
    adjustment_factor = 0.25
    return value - adjustment_factor * (value - mean)
