# test_model.py

import unittest
import os
import pandas as pd

from src.house_price_model import load_data, preprocess_data, train_model


class TestHousePriceModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.join("data", "Kc_house_data_NaN.csv")
        cls.df = load_data(cls.data_path)

    def test_data_loaded(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertTrue(len(self.df) > 0)

    def test_preprocessing(self):
        X, y = preprocess_data(self.df)
        self.assertEqual(len(X), len(y))
        self.assertTrue("price" not in X.columns)

    def test_model_training(self):
        X, y = preprocess_data(self.df)
        model, r2, mse = train_model(X, y)

        self.assertIsNotNone(model)
        self.assertTrue(-1 <= r2 <= 1)
        self.assertTrue(mse >= 0)


if __name__ == "__main__":
    unittest.main()
