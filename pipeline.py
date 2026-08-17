import pandas as pd
import glob
import matplotlib.pyplot as plt
 
def main():
    print("=== 1. ЗАГРУЗКА И ИЗУЧЕНИЕ ДАННЫХ (EDA) ===")
    
    # Ищем все 5 файлов транзакций
    transaction_files = glob.glob('daily_transactions_*.parquet')
    
    if not transaction_files:
        print("Файлы транзакций не найдены! Убедись, что они загружены в корень проекта.")
        return
 
    # Читаем и объединяем транзакции
    df_transactions = pd.concat([pd.read_parquet(f) for f in transaction_files], ignore_index=True)
    
    # Читаем файл инвентаря
    df_inventory = pd.read_parquet('product_inventory.parquet')
 
    # Рассказываем о данных (Tell me about the data)
    print("\n[Характеристики Transaction Data]")
    print(f"Количество строк и колонок: {df_transactions.shape}")
    print("\nТипы данных:")
    print(df_transactions.dtypes)
    
    print("\n[Характеристики Inventory Data]")
    print(f"Количество строк и колонок: {df_inventory.shape}")
 
    print("\n=== 2. ПРОЦЕССИНГ И АГРЕГАЦИЯ ===")
    
    # Объединяем транзакции и инвентарь по product_id
    df_merged = df_transactions.merge(df_inventory, on='product_id', how='left')
 
    # 1. Aggregate table of products sold by category
    # Суммируем количество (quantity) по категориям
    sold_by_category = df_merged.groupby('category')['quantity'].sum().reset_index()
    sold_by_category = sold_by_category.sort_values(by='quantity', ascending=False)
    print("\n--- Products Sold by Category ---")
    print(sold_by_category.to_string(index=False))
 
    # 2. Aggregate table of products sold by franchise
    # Суммируем количество (quantity) по франшизам
    sold_by_franchise = df_merged.groupby('franchise')['quantity'].sum().reset_index()
    sold_by_franchise = sold_by_franchise.sort_values(by='quantity', ascending=False)
    print("\n--- Products Sold by Franchise ---")
    print(sold_by_franchise.to_string(index=False))
 
    # 3. Table of transactions by user
    # Считаем уникальные transaction_id для каждого customer_id
    transactions_by_user = df_transactions.groupby('customer_id')['transaction_id'].nunique().reset_index()
    transactions_by_user.rename(columns={'transaction_id': 'total_transactions'}, inplace=True)
    transactions_by_user = transactions_by_user.sort_values(by='total_transactions', ascending=False)
    print("\n--- Transactions by User (Top 10 выведено для удобства) ---")
    print(transactions_by_user.head(10).to_string(index=False))
 
    print("\n=== 3. BONUS TASK ===")
    # Bonus: Top ten returning users based on their spend over the five days
    # Группируем по покупателю и суммируем потраченные деньги (line_total_usd)
    top_returning_users = df_transactions.groupby('customer_id')['line_total_usd'].sum().reset_index()
    top_returning_users = top_returning_users.sort_values(by='line_total_usd', ascending=False).head(10)
    top_returning_users.rename(columns={'line_total_usd': 'total_spend_usd'}, inplace=True)
    print("\n--- Top 10 Returning Users by Spend ---")
    print(top_returning_users.to_string(index=False))
 
    print("\n=== 4. ВИЗУАЛИЗАЦИЯ ===")
    # Graph showing total revenue for the day broken down by payment type
    
    # Создаем новую колонку 'date', извлекая только день из timestamp
    df_merged['date'] = pd.to_datetime(df_merged['timestamp']).dt.date
    
    # Агрегируем выручку по дате и методу оплаты
    revenue_by_payment = df_merged.groupby(['date', 'payment_method'])['line_total_usd'].sum().unstack()
 
    # Строим Stacked Bar Chart (столбчатая диаграмма с накоплением - отлично подходит для breakdown)
    revenue_by_payment.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    
    plt.title('Total Revenue per Day by Payment Method')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue (USD)')
    plt.xticks(rotation=45)
    plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Сохраняем график
    plt.savefig('revenue_by_payment.png')
    print("\n✅ График успешно сгенерирован и сохранен как 'revenue_by_payment.png'!")
 
if __name__ == "__main__":
    main()
 
