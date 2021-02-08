from enum import Enum
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ComplianceType(Enum):
    # constant compliance value for all items
    Single = 1
    # multiple compliance values, using the same features as treatment uplift values
    Multi = 2


class SyntheticDataGenerator:
    def __init__(self,
                 size: int,
                 compliances: List[float],
                 compliance_type: ComplianceType,
                 ctrl_outcome_rate: float,
                 treatment_uplifts: List[float],
                 n_splits: int,
                 seed: int):
        """
        individual features: X ∈ {1..n}
        treatment assigment: T ∈ {0, 1}
        mediation variable: M ∈ {1..n}
        outcome: Y ∈ {0, 1}
        random split ctrl/treat: P(T=1) = P(T=1|X=xi) = 0.5

        :param size: number of 'items': 'patients' in a medical context, 'users' in an advertising context
        :param compliances: P(M=1|T=1,X=xi)
                            ('drug intake' in a medical context, 'impression' in a advertising context)
        :param compliance_type:
        :param ctrl_outcome_rate: P(Y=1|T=0,X=xi) = P(Y=1[T=1,M=0,X=xi)
                                  ('self-healing' in a medical context, 'organic sale' in an advertising context)
        :param treatment_uplifts: (aka. ITET)  P(Y=1|T=1,M=1,X=xi) - P(Y=1|T=1,M=0,X=xi)
                                  ('intake effect' in a medical context, 'impression effect' in an advertising context)
        :param n_splits:
        :param seed:
        """
        if compliance_type == ComplianceType.Multi and len(treatment_uplifts) != len(compliances):
            raise ValueError("Not the same number of treatments uplifts and compliance values (ComplianceType.Multi)")
        self._size = size
        self._compliance_type = compliance_type

        # rates
        self._compliances = compliances[:]
        self._ctrl_outcome_rate = ctrl_outcome_rate
        #  P(Y=1|T=1,M=1,X=xi)
        self._treated_outcome_rates = np.clip([u + ctrl_outcome_rate for u in treatment_uplifts], 0, 1)

        self._n_splits = n_splits
        self._seed = seed
        self._features = [f"f{i}" for i in range(len(treatment_uplifts))]
        self._unique_items_cols = ['item_cat', 'uplift', 'T'] + self._features

    def get_total_n_splits(self):
        return len(self) * self._n_splits

    def split_in_tasks(self, n_cpus):
        # 'n' datasets x 's' splits
        assert n_cpus > 0
        n_datasets = len(self)
        if n_datasets >= n_cpus:
            rest = n_datasets % n_cpus
            tasks = [(i, np.arange(self._n_splits)) for i in range(n_datasets - rest)]
            if rest == 0:
                return tasks
            dataset_indices = np.arange(n_datasets - rest, n_datasets)
        else:
            tasks = []
            dataset_indices = np.arange(n_datasets)

        n_split_groups = n_cpus // len(dataset_indices)
        if n_cpus % len(dataset_indices) != 0:
            n_split_groups += 1
        split_groups = np.array_split(np.arange(self._n_splits), n_split_groups)

        tasks.extend((i_data, split_group) for i_data in dataset_indices for split_group in split_groups)
        return tasks

    def __len__(self):
        if self._compliance_type == ComplianceType.Single:
            return len(self._compliances)
        else:
            return 1

    def __getitem__(self, item: int) -> Tuple[str, 'Splitter']:
        if self._compliance_type == ComplianceType.Single:
            compliance = self._compliances[item]
            np.random.seed(self._seed + item)
            df = self._generate_data(compliance)
            return f'-{compliance}', Splitter(df, self._n_splits, self._unique_items_cols, self._features)
        else:
            assert item == 0
            np.random.seed(self._seed)
            df = self._generate_data(None)
            return '', Splitter(df, self._n_splits, self._unique_items_cols, self._features)

    def _generate_data(self, compliance):
        n_item_categories = len(self._treated_outcome_rates)

        df = pd.DataFrame()
        df['weight'] = np.ones(self._size, dtype=int)
        df['item_cat'] = np.random.randint(0, n_item_categories, size=self._size)
        q_1s = df['item_cat'].replace(to_replace=dict(enumerate(self._treated_outcome_rates))).values
        if compliance is None:  # ComplianceType.Multi
            compliance = df['item_cat'].replace(to_replace=dict(enumerate(self._compliances))).values

        df['uplift'] = compliance * (q_1s - self._ctrl_outcome_rate)  # P(Y=1|T=1,X=xi) - P(Y=1|T=0,X=xi)

        df['T'] = np.random.binomial(1, 0.5, self._size)
        df['M'] = (np.random.rand(self._size) < compliance).astype(int) * df['T']
        outcome_if_not_treated = np.random.binomial(1, self._ctrl_outcome_rate, self._size)
        outcome_if_treated = (np.random.rand(len(q_1s)) < q_1s).astype(int)
        outcome_t = df['T'] * (df['M'] * outcome_if_treated + (1 - df['M']) * outcome_if_not_treated)
        outcome_c = (1 - df['T']) * outcome_if_not_treated
        df['Y'] = outcome_t + outcome_c
        df['th_Y'] = df['uplift'] * (2 * df['T'] - 1) / 2  # theoretical outcome (for the 'AUUC_thout' metric)

        # create features
        for i in range(n_item_categories):
            df[f'f{i}'] = (df['item_cat'] == i).astype(int)
        return df


class Splitter:
    def __init__(self, df, n_splits, unique_items_cols, features):
        self._df = df
        self._n_splits = n_splits
        self._unique_items_cols = unique_items_cols
        self._features = features

    def get_features(self):
        return self._features

    def __len__(self):
        return self._n_splits

    def __getitem__(self, split):
        assert 0 <= split < self._n_splits
        train, test = train_test_split(self._df, test_size=0.5, random_state=split)
        # Convert test set to unique items (i.e. item categories) to speed-up the prediction.
        # If we have 'c' item categories and 'g' compliance values, we have '2 * c * g' unique items.
        # ('2 *' because treatment/control).
        test = test.groupby(self._unique_items_cols)\
            .agg({'Y': 'mean', 'th_Y': 'mean', 'weight': 'sum'})\
            .reset_index()
        test['th_weight'] = test['weight'].sum() / len(test)  # theoretical weight (for the 'AUUC_thout' metric)
        return train, test
