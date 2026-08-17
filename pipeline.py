"""
Data Pipeline: Transaction and Inventory Analysis

This script loads daily transaction and product inventory data from parquet files,
performs exploratory data analysis (EDA), aggregations, and creates visualizations.

Requirements:
- pandas
- matplotlib
- pyarrow or fastparquet (for parquet support)

Author: Data Engineer
Date: 2026-08-17
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import sys
import os


def export_aggregate_tables(category_table, franchise_table, user_table, df_transactions):
    """
    Export aggregate tables to separate CSV files.
    
    Args:
        category_table (pd.DataFrame): Products sold by category
        franchise_table (pd.DataFrame): Products sold by franchise
        user_table (pd.DataFrame): Transactions by user
        df_transactions (pd.DataFrame): Transaction data
    """
    try:
        print("\n[EXPORTING] Creating separate table files...")
        
        # 1. Export Products Sold by Category
        category_export = category_table.copy()
        category_export.to_csv('aggregate_products_by_category.csv', index=False)
        print(f"✓ Saved: aggregate_products_by_category.csv ({len(category_export)} rows)")
        
        # 2. Export Products Sold by Franchise
        franchise_export = franchise_table.copy()
        franchise_export.to_csv('aggregate_products_by_franchise.csv', index=False)
        print(f"✓ Saved: aggregate_products_by_franchise.csv ({len(franchise_export)} rows)")
        
        # 3. Export Transactions by User (All customers)
        user_export = user_table.copy()
        user_export.to_csv('aggregate_transactions_by_user.csv', index=False)
        print(f"✓ Saved: aggregate_transactions_by_user.csv ({len(user_export)} rows)")
        
        # 4. Export Top 10 Returning Users by Spend
        top_users = df_transactions.groupby('customer_id')['line_total_usd'].sum().reset_index()
        top_users.columns = ['Customer_ID', 'Total_Spend_USD']
        top_users = top_users.sort_values(by='Total_Spend_USD', ascending=False).head(10)
        top_users['Rank'] = range(1, len(top_users) + 1)
        top_users = top_users[['Rank', 'Customer_ID', 'Total_Spend_USD']]
        top_users.to_csv('aggregate_top_10_users_by_spend.csv', index=False)
        print(f"✓ Saved: aggregate_top_10_users_by_spend.csv (10 rows)")
        
        # 5. Export Revenue Summary
        revenue_summary = pd.DataFrame({
            'Metric': [
                'Total Revenue (USD)',
                'Total Units Sold',
                'Total Transactions',
                'Unique Customers',
                'Average Transaction Value (USD)',
                'Average Customer Spend (USD)'
            ],
            'Value': [
                f"{df_transactions['line_total_usd'].sum():,.2f}",
                f"{df_transactions['quantity'].sum():,}",
                f"{len(df_transactions):,}",
                f"{df_transactions['customer_id'].nunique():,}",
                f"{df_transactions['line_total_usd'].mean():,.2f}",
                f"{df_transactions.groupby('customer_id')['line_total_usd'].sum().mean():,.2f}"
            ]
        })
        revenue_summary.to_csv('aggregate_revenue_summary.csv', index=False)
        print(f"✓ Saved: aggregate_revenue_summary.csv (summary metrics)")
        
        print("\n📁 All table files exported to workspace root!")
        
    except Exception as e:
        print(f"❌ Error exporting tables: {e}")


def detect_franchise_column(df_inventory):
    """
    Dynamically detect the franchise/brand column in the inventory DataFrame.
    
    Args:
        df_inventory (pd.DataFrame): The inventory data
        
    Returns:
        str: The name of the franchise column, or None if not found
    """
    # List of common franchise column names
    possible_names = ['franchise', 'franchise_theme', 'brand', 'Franchise', 'Brand', 'FRANCHISE', 'BRAND']
    
    for col_name in possible_names:
        if col_name in df_inventory.columns:
            print(f"✓ Detected franchise column: '{col_name}'")
            return col_name
    
    # If no standard name found, print warning
    print("⚠ Warning: No franchise/brand column detected with standard names.")
    print(f"   Available columns: {df_inventory.columns.tolist()}")
    return None


def load_data():
    """
    Load transaction and inventory data from parquet files.
    
    Returns:
        tuple: (df_transactions, df_inventory) or (None, None) if loading fails
    """
    try:
        # Load all transaction files
        transaction_files = sorted(glob.glob('daily_transactions_*.parquet'))
        
        if not transaction_files:
            print("❌ Error: No transaction files found! Expected files like 'daily_transactions_YYYY-MM-DD.parquet'")
            return None, None
        
        print(f"Loading {len(transaction_files)} transaction file(s)...")
        df_transactions = pd.concat(
            [pd.read_parquet(f) for f in transaction_files],
            ignore_index=True
        )
        
        # Load inventory file
        if not os.path.exists('product_inventory.parquet'):
            print("❌ Error: product_inventory.parquet not found!")
            return None, None
        
        print("Loading inventory file...")
        df_inventory = pd.read_parquet('product_inventory.parquet')
        
        return df_transactions, df_inventory
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None


def perform_eda(df_transactions, df_inventory):
    """
    Perform Exploratory Data Analysis on transaction and inventory data.
    
    Args:
        df_transactions (pd.DataFrame): Transaction data
        df_inventory (pd.DataFrame): Inventory data
    """
    print("\n" + "="*80)
    print("1. EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Transaction Data Characteristics
    print("\n[TRANSACTION DATA - What We Found]")
    print(f"  Shape: {df_transactions.shape[0]:,} rows × {df_transactions.shape[1]} columns")
    print(f"  Date Range: {df_transactions['timestamp'].min()} to {df_transactions['timestamp'].max()}")
    print(f"  Unique Customers: {df_transactions['customer_id'].nunique():,}")
    print(f"  Unique Products: {df_transactions['product_id'].nunique():,}")
    print(f"  Total Revenue: ${df_transactions['line_total_usd'].sum():,.2f}")
    print(f"  Total Units Sold: {df_transactions['quantity'].sum():,}")
    print(f"  Missing Values:\n{df_transactions.isnull().sum()}")
    print(f"\n  Data Types:\n{df_transactions.dtypes}")
    
    # Inventory Data Characteristics
    print("\n[INVENTORY DATA - What We Found]")
    print(f"  Shape: {df_inventory.shape[0]:,} rows × {df_inventory.shape[1]} columns")
    print(f"  Active Products: {df_inventory['active'].sum():,}")
    print(f"  Total Stock Value: ${(df_inventory['stock_quantity'] * df_inventory['unit_cost_usd']).sum():,.2f}")
    print(f"  Missing Values:\n{df_inventory.isnull().sum()}")
    print(f"\n  Data Types:\n{df_inventory.dtypes}")


def process_and_aggregate(df_transactions, df_inventory):
    """
    Perform data processing and create aggregation tables.
    Filters for only COMPLETED transactions.
    
    Args:
        df_transactions (pd.DataFrame): Transaction data
        df_inventory (pd.DataFrame): Inventory data
        
    Returns:
        tuple: (df_merged, franchise_column_name)
    """
    print("\n" + "="*80)
    print("2. DATA PROCESSING & AGGREGATION (COMPLETED TRANSACTIONS ONLY)")
    print("="*80)
    
    # Filter for completed transactions only
    total_transactions = len(df_transactions)
    df_transactions = df_transactions[df_transactions['order_status'] == 'Completed']
    completed_count = len(df_transactions)
    print(f"\n📊 Transaction Filter Applied:")
    print(f"   Total Transactions: {total_transactions:,}")
    print(f"   Completed Transactions: {completed_count:,}")
    print(f"   Filtered Out: {total_transactions - completed_count:,} ({100 * (total_transactions - completed_count) / total_transactions:.1f}%)")
    
    # Merge transactions and inventory
    try:
        df_merged = df_transactions.merge(df_inventory, on='product_id', how='left')
        print(f"\n✓ Successfully merged data: {df_merged.shape[0]:,} rows")
    except Exception as e:
        print(f"❌ Error merging data: {e}")
        return None, None
    
    # Detect franchise column dynamically
    franchise_col = detect_franchise_column(df_inventory)
    
    # Aggregation 1: Products Sold by Category
    print("\n[1] Products Sold by Category")
    print("-" * 80)
    sold_by_category = df_merged.groupby('category', dropna=False)['quantity'].sum().reset_index()
    sold_by_category.columns = ['Category', 'Total_Quantity_Sold']
    sold_by_category = sold_by_category.sort_values(by='Total_Quantity_Sold', ascending=False)
    print(sold_by_category.to_string(index=False))
    
    # Aggregation 2: Products Sold by Franchise (if available)
    if franchise_col:
        print("\n[2] Products Sold by Franchise/Brand")
        print("-" * 80)
        sold_by_franchise = df_merged.groupby(franchise_col, dropna=False)['quantity'].sum().reset_index()
        sold_by_franchise.columns = ['Franchise_Theme', 'Total_Quantity_Sold']
        sold_by_franchise = sold_by_franchise.sort_values(by='Total_Quantity_Sold', ascending=False)
        print(sold_by_franchise.to_string(index=False))
    else:
        print("\n[2] Products Sold by Franchise/Brand")
        print("-" * 80)
        print("⚠ Franchise column not available for aggregation")
    
    # Aggregation 3: Transactions by User (Count of unique transactions per customer)
    print("\n[3] Transactions by User")
    print("-" * 80)
    transactions_by_user = df_transactions.groupby('customer_id')['transaction_id'].nunique().reset_index()
    transactions_by_user.columns = ['Customer_ID', 'Unique_Transactions']
    transactions_by_user = transactions_by_user.sort_values(by='Unique_Transactions', ascending=False)
    print(f"Total unique customers: {len(transactions_by_user):,}")
    print("\nTop 10 customers by transaction count:")
    print(transactions_by_user.head(10).to_string(index=False))
    
    # Export tables to separate CSV files
    export_aggregate_tables(sold_by_category, sold_by_franchise, transactions_by_user, df_transactions)
    
    return df_merged, franchise_col


def bonus_top_returning_users(df_transactions):
    """
    Bonus Task: Identify top ten returning users based on spend.
    
    Args:
        df_transactions (pd.DataFrame): Transaction data
    """
    print("\n" + "="*80)
    print("3. BONUS: TOP 10 RETURNING USERS BY TOTAL SPEND")
    print("="*80)
    
    top_users = df_transactions.groupby('customer_id')['line_total_usd'].sum().reset_index()
    top_users.columns = ['Customer_ID', 'Total_Spend_USD']
    top_users = top_users.sort_values(by='Total_Spend_USD', ascending=False).head(10)
    top_users['Rank'] = range(1, len(top_users) + 1)
    top_users = top_users[['Rank', 'Customer_ID', 'Total_Spend_USD']]
    top_users['Total_Spend_USD'] = top_users['Total_Spend_USD'].apply(lambda x: f"${x:,.2f}")
    
    print(top_users.to_string(index=False))


def create_visualizations(df_merged):
    """
    Create multiple visualizations for comprehensive data analysis.
    
    Args:
        df_merged (pd.DataFrame): Merged transaction and inventory data
    """
    print("\n" + "="*80)
    print("4. VISUALIZATIONS (MULTIPLE GRAPHICS)")
    print("="*80)
    
    try:
        # Extract date from timestamp
        df_merged['date'] = pd.to_datetime(df_merged['timestamp']).dt.date
        
        # Graph 1: Revenue by Payment Method (Stacked Bar)
        print("\n[Graph 1] Creating: Revenue by Payment Method...")
        revenue_by_payment = df_merged.groupby(['date', 'payment_method'], dropna=False)['line_total_usd'].sum().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        revenue_by_payment.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            colormap='Set3',
            width=0.7
        )
        
        plt.title('Total Daily Revenue by Payment Method', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Total Revenue (USD)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Payment Method', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('revenue_by_payment.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'revenue_by_payment.png'")
        
        # Graph 2: Top 10 Products by Revenue
        print("\n[Graph 2] Creating: Top 10 Products by Revenue...")
        top_products = df_merged.groupby('product_name')['line_total_usd'].sum().nlargest(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(top_products)), top_products['line_total_usd'], color='steelblue')
        ax.set_yticks(range(len(top_products)))
        ax.set_yticklabels(top_products['product_name'], fontsize=10)
        ax.set_xlabel('Total Revenue (USD)', fontsize=12)
        ax.set_title('Top 10 Products by Revenue', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (idx, row) in enumerate(top_products.iterrows()):
            ax.text(row['line_total_usd'], i, f"  ${row['line_total_usd']:,.0f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('top_10_products_by_revenue.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'top_10_products_by_revenue.png'")
        
        # Graph 3: Category Revenue Distribution (Pie Chart)
        print("\n[Graph 3] Creating: Category Revenue Distribution...")
        category_revenue = df_merged.groupby('category')['line_total_usd'].sum()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(category_revenue)))
        wedges, texts, autotexts = ax.pie(
            category_revenue,
            labels=category_revenue.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 10}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
        
        plt.title('Revenue Distribution by Category', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('revenue_by_category_pie.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'revenue_by_category_pie.png'")
        
        # Graph 4: Top 10 Franchises by Revenue (Bar Chart)
        print("\n[Graph 4] Creating: Top 10 Franchises by Revenue...")
        top_franchises = df_merged.groupby('franchise_theme')['line_total_usd'].sum().nlargest(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(top_franchises)), top_franchises['line_total_usd'], color='coral')
        ax.set_xticks(range(len(top_franchises)))
        ax.set_xticklabels(top_franchises['franchise_theme'], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Total Revenue (USD)', fontsize=12)
        ax.set_title('Top 10 Franchises by Revenue', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(top_franchises['line_total_usd']):
            ax.text(i, v, f"${v/1e6:.2f}M", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('top_10_franchises_by_revenue.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'top_10_franchises_by_revenue.png'")
        
        # Graph 5: Daily Revenue Trend
        print("\n[Graph 5] Creating: Daily Revenue Trend...")
        daily_revenue = df_merged.groupby('date')['line_total_usd'].sum().reset_index()
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_revenue['date'], daily_revenue['line_total_usd'], marker='o', linewidth=2.5, 
                markersize=8, color='darkgreen')
        ax.fill_between(daily_revenue['date'], daily_revenue['line_total_usd'], alpha=0.3, color='lightgreen')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Revenue (USD)', fontsize=12)
        ax.set_title('Daily Revenue Trend', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for idx, row in daily_revenue.iterrows():
            ax.text(row['date'], row['line_total_usd'], f"${row['line_total_usd']/1e6:.2f}M", 
                   ha='center', va='bottom', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('daily_revenue_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'daily_revenue_trend.png'")
        
        # Graph 6: Device Type Revenue Distribution (Bar Chart)
        print("\n[Graph 6] Creating: Revenue by Device Type...")
        device_revenue = df_merged.groupby('device_type')['line_total_usd'].sum().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(device_revenue)), device_revenue.values, color='mediumpurple')
        ax.set_xticks(range(len(device_revenue)))
        ax.set_xticklabels(device_revenue.index, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Total Revenue (USD)', fontsize=12)
        ax.set_title('Revenue by Device Type', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(device_revenue.values):
            ax.text(i, v, f"${v/1e6:.2f}M", ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('revenue_by_device_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'revenue_by_device_type.png'")
        
        # Graph 7: Customer Spending Distribution (Histogram)
        print("\n[Graph 7] Creating: Customer Spending Distribution...")
        customer_spending = df_merged.groupby('customer_id')['line_total_usd'].sum()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        n, bins, patches = ax.hist(customer_spending, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(customer_spending.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${customer_spending.mean():,.2f}')
        ax.axvline(customer_spending.median(), color='green', linestyle='--', linewidth=2, label=f'Median: ${customer_spending.median():,.2f}')
        
        ax.set_xlabel('Total Spending per Customer (USD)', fontsize=12)
        ax.set_ylabel('Number of Customers', fontsize=12)
        ax.set_title('Customer Spending Distribution (5-Day Period)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customer_spending_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved as 'customer_spending_distribution.png'")
        
        print("\n✅ All visualizations created successfully!")
        print("\nGenerated Files:")
        print("  1. revenue_by_payment.png - Daily revenue by payment method")
        print("  2. top_10_products_by_revenue.png - Top products")
        print("  3. revenue_by_category_pie.png - Category revenue distribution")
        print("  4. top_10_franchises_by_revenue.png - Top franchises")
        print("  5. daily_revenue_trend.png - Revenue trend over time")
        print("  6. revenue_by_device_type.png - Revenue by device type")
        print("  7. customer_spending_distribution.png - Customer spending patterns")
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {e}")


def main():
    """
    Main pipeline execution function.
    """
    print("\n" + "="*80)
    print("DATA PIPELINE: Transaction and Inventory Analysis")
    print("="*80)
    
    # Step 1: Load data
    df_transactions, df_inventory = load_data()
    if df_transactions is None or df_inventory is None:
        print("\n❌ Pipeline failed at data loading step.")
        sys.exit(1)
    
    # Step 2: Perform EDA
    perform_eda(df_transactions, df_inventory)
    
    # Step 3: Process and aggregate
    df_merged, franchise_col = process_and_aggregate(df_transactions, df_inventory)
    if df_merged is None:
        print("\n❌ Pipeline failed at data processing step.")
        sys.exit(1)
    
    # Step 4: Bonus - Top returning users
    bonus_top_returning_users(df_transactions)
    
    # Step 5: Create multiple visualizations
    create_visualizations(df_merged)
    
    print("\n" + "="*80)
    print("✓ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
 
