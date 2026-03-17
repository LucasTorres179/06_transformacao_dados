import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('clientes-v2-tratados.csv')

print(df.head())

df = df[['idade', 'salario']]

# Normalização - MinMaxScaler (0 a 1)
scaler = MinMaxScaler()
df['idadeMinMaxScaler'] = scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler'] = scaler.fit_transform(df[['salario']])

# Normalização - MinMaxScaler (-1 a 1)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
df['idadeMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['idade']])
df['salarioMinMaxScaler_mm'] = min_max_scaler.fit_transform(df[['salario']])

# Padronização - StandardScaler
scaler = StandardScaler()
df['idadeStandardScaler'] = scaler.fit_transform(df[['idade']])
df['salarioStandardScaler'] = scaler.fit_transform(df[['salario']])

# Padronização - RobustScaler
scaler = RobustScaler()
df['idadeRobustScaler'] = scaler.fit_transform(df[['idade']])
df['salarioRobustScaler'] = scaler.fit_transform(df[['salario']])

print(df.head(15))

print("MinMaxScaler (De 0 a 1)")
print("Idade - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeMinMaxScaler'].min(),
    df['idadeMinMaxScaler'].mean(),
    df['idadeMinMaxScaler'].std()
))
print("Salário - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['salarioMinMaxScaler'].min(),
    df['salarioMinMaxScaler'].mean(),
    df['salarioMinMaxScaler'].std()
))

print("\nMinMaxScaler (De -1 a 1)")
print("Idade - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeMinMaxScaler_mm'].min(),
    df['idadeMinMaxScaler_mm'].mean(),
    df['idadeMinMaxScaler_mm'].std()
))
print("Salário - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['salarioMinMaxScaler_mm'].min(),
    df['salarioMinMaxScaler_mm'].mean(),
    df['salarioMinMaxScaler_mm'].std()
))

print("\nStandardScaler (Ajuste a média a 0 e desvio a 1)")
print("Idade - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['idadeStandardScaler'].min(),
    df['idadeStandardScaler'].mean(),
    df['idadeStandardScaler'].std()
))
print("Salário - Min: {:.4f} Mean: {:.4f} Std: {:.4f}".format(
    df['salarioStandardScaler'].min(),
    df['salarioStandardScaler'].mean(),
    df['salarioStandardScaler'].std()
))