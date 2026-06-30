#!/usr/bin/env python3
"""
Script to filter and separate questions from qa_gold.csv into separate files by type.
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_and_separate_questions(input_csv, output_dir=None, filter_by_type=None):
    """
    Filter questions from CSV and separate them into different files.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output files (default: same as input)
        filter_by_type: List of types to filter by, e.g., ['Fact', 'Reasoning']
                       If None, separates by all types found
    """
    try:
        # Read the CSV file
        logger.info(f"Reading CSV file: {input_csv}")
        df = pd.read_csv(input_csv)
        logger.info(f"Total rows: {len(df)}")
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(input_csv).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get unique types
        if 'type' not in df.columns:
            logger.error("Column 'type' not found in CSV. Available columns: {}", df.columns.tolist())
            return False
        
        if 'question' not in df.columns:
            logger.error("Column 'question' not found in CSV. Available columns: {}", df.columns.tolist())
            return False
        
        all_types = df['type'].unique()
        logger.info(f"Found types: {all_types}")
        
        # Filter types if specified
        types_to_process = filter_by_type if filter_by_type else list(all_types)
        
        # Separate and save by type
        for question_type in types_to_process:
            if question_type not in all_types:
                logger.warning(f"Type '{question_type}' not found in data")
                continue
            
            filtered_df = df[df['type'] == question_type]
            # Extract only the question column
            questions_only = filtered_df[['question']].reset_index(drop=True)
            output_file = output_dir / f"qa_gold_{question_type.lower()}.csv"
            
            questions_only.to_csv(output_file, index=False)
            logger.info(f"Saved {len(questions_only)} {question_type} questions to: {output_file}")
        
        # Also save a summary
        summary_file = output_dir / "qa_gold_summary.csv"
        summary_df = df.groupby('type').size().reset_index(name='count')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Summary saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False


def filter_by_criteria(input_csv, output_csv, criteria_dict):
    """
    Filter CSV based on custom criteria.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        criteria_dict: Dictionary with column names as keys and filter values
                      e.g., {'type': ['Fact', 'Reasoning'], 'run_tag': 'final_exam_v1'}
    """
    try:
        logger.info(f"Reading CSV file: {input_csv}")
        df = pd.read_csv(input_csv)
        
        # Apply filters
        for column, values in criteria_dict.items():
            if column not in df.columns:
                logger.warning(f"Column '{column}' not found in CSV")
                continue
            
            if isinstance(values, list):
                df = df[df[column].isin(values)]
            else:
                df = df[df[column] == values]
            
            logger.info(f"Filtered by {column}={values}: {len(df)} rows remaining")
        
        # Save filtered results
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved {len(df)} filtered questions to: {output_csv}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import sys
    
    # Get paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "backend" / "media" / "rag_chatbot.qa_gold.csv"
    output_dir = script_dir / "backend" / "media" / "separated_questions"
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print("Usage:")
            print("  python filter_and_separate_questions.py")
            print("    Separates all questions by type")
            print()
            print("  python filter_and_separate_questions.py <input_csv> <output_dir>")
            print("    Separates questions from input_csv to output_dir")
            sys.exit(0)
        elif len(sys.argv) >= 2:
            input_csv = Path(sys.argv[1])
            if len(sys.argv) >= 3:
                output_dir = Path(sys.argv[2])
    
    if not input_csv.exists():
        logger.error(f"Input file not found: {input_csv}")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Question Filtering and Separation Script")
    logger.info("=" * 60)
    
    # Method 1: Separate by type
    success = filter_and_separate_questions(str(input_csv), str(output_dir))
    
    if success:
        logger.info("=" * 60)
        logger.info("Process completed successfully!")
        logger.info("=" * 60)
        
        # Example: Filter by specific criteria
        logger.info("\nExample of filtering by criteria:")
        example_output = output_dir / "qa_gold_fact_final_exam.csv"
        filter_by_criteria(
            str(input_csv),
            str(example_output),
            {'type': 'Fact', 'run_tag': 'final_exam_v1'}
        )
    else:
        logger.error("Process failed!")
        sys.exit(1)
