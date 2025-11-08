
import os
import json
from google.cloud import spanner
from google.api_core import exceptions

# Configuration - try to load from env, fallback to defaults seen in logs
PROJECT_ID = os.environ.get("GRAPHRAG_PROJECT_ID") or os.environ.get("GCP_PROJECT_ID") or "cloud-llm-preview1"
INSTANCE_ID = os.environ.get("GRAPHRAG_INSTANCE_ID") or os.environ.get("SPANNER_INSTANCE_ID") or "graphrag-instance"
DATABASE_ID = os.environ.get("GRAPHRAG_DATABASE_ID") or os.environ.get("SPANNER_DATABASE_ID") or "graphrag"
TABLE_PREFIX = os.environ.get("GRAPHRAG_TABLE_PREFIX") or ""

print(f"--- Spanner Configuration ---")
print(f"Project:  {PROJECT_ID}")
print(f"Instance: {INSTANCE_ID}")
print(f"Database: {DATABASE_ID}")
print(f"Prefix:   '{TABLE_PREFIX}'")
print("-----------------------------")

spanner_client = spanner.Client(project=PROJECT_ID)
instance = spanner_client.instance(INSTANCE_ID)
database = instance.database(DATABASE_ID)

def check_table_schema(table_name_suffix):
    table_name = f"{TABLE_PREFIX}{table_name_suffix}"
    print(f"Checking Schema for table: {table_name}")
    try:
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                "SELECT COLUMN_NAME, SPANNER_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = @table_name",
                params={"table_name": table_name},
                param_types={"table_name": spanner.param_types.STRING}
            )
            rows = list(results)
            if not rows:
                print(f"  [WARNING] Table {table_name} NOT FOUND in INFORMATION_SCHEMA.")
                return False
            for row in rows:
                col_name, spanner_type = row
                # Highlight complex types we care about
                if "ARRAY" in spanner_type or "JSON" in spanner_type:
                     print(f"  - {col_name:<20} : {spanner_type}  <-- COMPLEX TYPE")
                else:
                     print(f"  - {col_name:<20} : {spanner_type}")
            return True
    except Exception as e:
        print(f"  [ERROR] Failed to check schema: {e}")
        return False

def check_data_sample(table_name_suffix, columns_to_check):
    table_name = f"{TABLE_PREFIX}{table_name_suffix}"
    print(f"Checking Data Samples for table: {table_name}")
    cols_str = ", ".join([f"`{c}`" for c in columns_to_check])
    try:
        with database.snapshot() as snapshot:
            # Get a few rows that ideally have non-empty data for these columns
            query = f"SELECT {cols_str} FROM `{table_name}` LIMIT 5"
            results = snapshot.execute_sql(query)
            rows = list(results)
            
            if not rows:
                 print("  [WARNING] Table is empty, cannot verify data types.")
                 return

            for i, row in enumerate(rows):
                print(f"  Row {i+1}:")
                for j, col_val in enumerate(row):
                    col_name = columns_to_check[j]
                    val_type = type(col_val).__name__
                    
                    # Special check for Spanner JSON objects
                    if val_type == 'JsonObject':
                        val_str = json.dumps(col_val)
                        print(f"    - {col_name:<20}: Type={val_type}, Value={val_str[:100]}...")
                    elif isinstance(col_val, list):
                        print(f"    - {col_name:<20}: Type=list (len={len(col_val)}), Value={str(col_val)[:100]}...")
                    else:
                        print(f"    - {col_name:<20}: Type={val_type}, Value={str(col_val)[:100]}...")
            print("")
    except Exception as e:
        print(f"  [ERROR] Failed to read data: {e}")

def main():
    # 1. Check communities (focus on 'children' which caused issues before)
    if check_table_schema("communities"):
        check_data_sample("communities", ["id", "children", "entity_ids"])

    print("-" * 40 + "\n")

    # 2. Check text_units (focus on ARRAYs)
    if check_table_schema("text_units"):
        check_data_sample("text_units", ["id", "document_ids", "entity_ids"])

    print("-" * 40 + "\n")

    # 3. Check entities (focus on text_unit_ids ARRAY)
    if check_table_schema("entities"):
        check_data_sample("entities", ["id", "title", "text_unit_ids"])

if __name__ == "__main__":
    main()
