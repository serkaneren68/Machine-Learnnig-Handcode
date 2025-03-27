{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Bisiklet Kiralama Veri Seti ile Regresyon Tabanlı SVM (SVR) Modeli\n",
        "\n",
        "Bu notebook'ta UCI Bisiklet Kiralama veri setini kullanarak regresyon tabanlı bir SVM modeli oluşturacak ve 6-fold cross validation ile değerlendireceğiz."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Gerekli Kütüphanelerin Yüklenmesi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.svm import SVR\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.model_selection import KFold\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.pipeline import Pipeline\n",
        "from ucimlrepo import fetch_ucirepo"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Veri Setinin Yüklenmesi\n",
        "\n",
        "UCI Machine Learning Repository'den bisiklet kiralama veri setini yükleyelim."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# Bisiklet kiralama veri setini UCI Repository'den yükleyelim\n",
        "bike_sharing = fetch_ucirepo(id=275)\n",
        "\n",
        "# Veri çerçevelerini alalım\n",
        "df = bike_sharing.data.features\n",
        "target = bike_sharing.data.targets\n",
        "\n",
        "# Hedef değişkeni seçelim ('cnt' toplam kiralama sayısı)\n",
        "target = target['cnt']\n",
        "\n",
        "print(f\"Yüklenen veri boyutu: {df.shape}\")\n",
        "df.head()"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Veri Ön İşleme\n",
        "\n",
        "Kategorik değişkenleri one-hot encoding ile dönüştürelim ve veriyi modelleme için hazırlayalım."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# Kategorik değişkenleri one-hot encoding ile dönüştürelim\n",
        "categorical_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']\n",
        "df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
        "\n",
        "# Tarih sütununu kaldıralım (zaten diğer özellikler ile temsil edilmiş)\n",
        "if 'dteday' in df.columns:\n",
        "    df = df.drop('dteday', axis=1)\n",
        "\n",
        "print(f\"Ön işleme sonrası özellik sayısı: {df.shape[1]}\")\n",
        "df.head()"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Veri Setini Hazırlama"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# Veri setini hazırlayalım\n",
        "X = df.values\n",
        "y = target.values\n",
        "\n",
        "print(f\"X şekli: {X.shape}\")\n",
        "print(f\"y şekli: {y.shape}\")"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 6-Fold Cross Validation ile SVR Modelinin Değerlendirilmesi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# 6-fold cross validation uygulayalım\n",
        "kf = KFold(n_splits=6, shuffle=True, random_state=42)\n",
        "\n",
        "# SVR modelini hazırlayalım - Pipeline ile ölçeklendirme ekliyoruz\n",
        "pipeline = Pipeline([\n",
        "    ('scaler', StandardScaler()),\n",
        "    ('svr', SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.1))\n",
        "])\n",
        "\n",
        "# MSE metriğini saklayacağımız liste\n",
        "mse_scores = []\n",
        "\n",
        "plt.figure(figsize=(18, 12))\n",
        "\n",
        "fold = 1\n",
        "# Cross-validation\n",
        "for train_idx, test_idx in kf.split(X):\n",
        "    X_train, X_test = X[train_idx], X[test_idx]\n",
        "    y_train, y_test = y[train_idx], y[test_idx]\n",
        "    \n",
        "    # Modeli eğitelim\n",
        "    pipeline.fit(X_train, y_train)\n",
        "    \n",
        "    # Test seti üzerinde tahminleri yapalım\n",
        "    y_pred = pipeline.predict(X_test)\n",
        "    \n",
        "    # MSE metriğini hesaplayalım\n",
        "    mse = mean_squared_error(y_test, y_pred)\n",
        "    mse_scores.append(mse)\n",
        "    \n",
        "    # Tahmin vs gerçek değer grafiğini çizelim\n",
        "    plt.subplot(2, 3, fold)\n",
        "    plt.scatter(y_test, y_pred, alpha=0.5)\n",
        "    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)\n",
        "    plt.xlabel('Gerçek Değerler')\n",
        "    plt.ylabel('Tahminler')\n",
        "    plt.title(f'Fold {fold} (MSE = {mse:.2f})')\n",
        "    \n",
        "    fold += 1\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Tüm Fold'lar için Birleşik Grafik"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# Tüm fold'ların birleşik grafiğini çizelim\n",
        "plt.figure(figsize=(12, 8))\n",
        "all_y_test = []\n",
        "all_y_pred = []\n",
        "\n",
        "for train_idx, test_idx in kf.split(X):\n",
        "    X_train, X_test = X[train_idx], X[test_idx]\n",
        "    y_train, y_test = y[train_idx], y[test_idx]\n",
        "    \n",
        "    pipeline.fit(X_train, y_train)\n",
        "    y_pred = pipeline.predict(X_test)\n",
        "    \n",
        "    all_y_test.extend(y_test)\n",
        "    all_y_pred.extend(y_pred)\n",
        "\n",
        "plt.scatter(all_y_test, all_y_pred, alpha=0.3)\n",
        "plt.plot([min(all_y_test), max(all_y_test)], [min(all_y_test), max(all_y_test)], 'k--', lw=2)\n",
        "plt.xlabel('Gerçek Değerler')\n",
        "plt.ylabel('Tahminler')\n",
        "plt.title('Tüm Fold\\'lar için Gerçek vs Tahmin')\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Sonuçların Değerlendirilmesi"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "source": [
        "# Sonuçları yazdıralım\n",
        "print(\"Regresyon Metrikleri:\")\n",
        "print(f\"Ortalama MSE: {np.mean(mse_scores):.2f} ± {np.std(mse_scores):.2f}\")\n",
        "\n",
        "# Her fold için sonuçları ayrı ayrı yazdıralım\n",
        "print(\"\\nHer fold için MSE:\")\n",
        "for i in range(6):\n",
        "    print(f\"Fold {i+1} - MSE: {mse_scores[i]:.2f}\")"
      ],
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Sonuç\n",
        "\n",
        "Bu notebook'ta UCI Bisiklet Kiralama veri setini kullanarak SVR modeli eğittik ve 6-fold cross validation ile değerlendirdik. Her fold için MSE değerlerini hesapladık ve tahminleri görselleştirdik."
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}