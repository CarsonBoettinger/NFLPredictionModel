"""
NFL Player Statistics Data Pipeline

This script fetches NFL player statistics using nfl_data_py, preprocesses the data,
and saves it locally for machine learning applications.

Dependencies:
pip install nfl_data_py pandas numpy scikit-learn pyarrow

Author: Carson Boettinger
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nfl_data_py as nfl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_data_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NFLDataPipeline:
    """Main class for NFL data fetching, processing, and feature engineering."""
    
    def __init__(self, data_dir: str = "nfl_data"):
        """
        Initialize the NFL data pipeline.
        
        Args:
            data_dir: Directory to save data files
        """
        self.data_dir = data_dir
        self.raw_data_path = os.path.join(data_dir, "raw")
        self.processed_data_path = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fetch_player_stats(self, years: List[int] = None, stat_types: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch NFL player statistics from nfl_data_py.
        
        Args:
            years: List of years to fetch (default: 1999-2023)
            stat_types: Types of stats to fetch
            
        Returns:
            Dictionary of DataFrames containing different stat types
        """
        if years is None:
            years = list(range(1999, 2024))  # nfl_data_py goes back to 1999
            
        if stat_types is None:
            stat_types = ['passing', 'rushing', 'receiving']
        
        logger.info(f"Fetching NFL player stats for years: {min(years)}-{max(years)}")
        
        all_stats = {}
        
        try:
            # Get all seasonal data at once - nfl_data_py returns comprehensive stats
            logger.info("Fetching comprehensive seasonal statistics...")
            
            # Try to fetch data, if it fails try without the most recent year
            try:
                all_seasonal_data = nfl.import_seasonal_data(years, s_type='REG')
            except Exception as e:
                if "404" in str(e) and len(years) > 1:
                    logger.warning(f"Most recent year ({max(years)}) not available, trying without it...")
                    years = years[:-1]  # Remove the most recent year
                    all_seasonal_data = nfl.import_seasonal_data(years, s_type='REG')
                else:
                    raise
            
            if not all_seasonal_data.empty:
                logger.info(f"Fetched {len(all_seasonal_data)} total seasonal records")
                logger.info(f"Available columns: {list(all_seasonal_data.columns)}")
                
                # Split into different stat types based on available columns
                for stat_type in stat_types:
                    if stat_type == 'passing':
                        # Check for various passing column names
                        pass_cols = ['pass_att', 'passing_att', 'att', 'attempts']
                        pass_col = None
                        for col in pass_cols:
                            if col in all_seasonal_data.columns:
                                pass_col = col
                                break
                        
                        if pass_col and all_seasonal_data[pass_col].sum() > 0:
                            stats_df = all_seasonal_data[all_seasonal_data[pass_col] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                            
                    elif stat_type == 'rushing':
                        # Check for various rushing column names
                        rush_cols = ['rush_att', 'rushing_att', 'carries', 'car']
                        rush_col = None
                        for col in rush_cols:
                            if col in all_seasonal_data.columns:
                                rush_col = col
                                break
                        
                        if rush_col and all_seasonal_data[rush_col].sum() > 0:
                            stats_df = all_seasonal_data[all_seasonal_data[rush_col] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                            
                    elif stat_type == 'receiving':
                        # Check for various receiving column names
                        rec_cols = ['rec', 'receptions', 'catches']
                        rec_col = None
                        for col in rec_cols:
                            if col in all_seasonal_data.columns:
                                rec_col = col
                                break
                        
                        if rec_col and all_seasonal_data[rec_col].sum() > 0:
                            stats_df = all_seasonal_data[all_seasonal_data[rec_col] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                    else:
                        logger.warning(f"Unknown stat type: {stat_type}")
                        continue
                    
                    if not stats_df.empty:
                        all_stats[stat_type] = stats_df
                        logger.info(f"Filtered {len(stats_df)} {stat_type} records")
                    else:
                        logger.warning(f"No data found for {stat_type} - checked columns: {[col for col in all_seasonal_data.columns if stat_type[:4] in col.lower()]}")
            else:
                logger.warning("No seasonal data retrieved")
                    
        except Exception as e:
            logger.error(f"Error fetching player stats: {str(e)}")
            raise
        
        # Also fetch roster data for player metadata
        try:
            logger.info("Fetching roster data for player metadata...")
            # Try different roster function names
            if hasattr(nfl, 'import_rosters'):
                roster_data = nfl.import_rosters(years)
            elif hasattr(nfl, 'import_roster_data'):
                roster_data = nfl.import_roster_data(years)
            else:
                logger.warning("No roster import function found")
                roster_data = pd.DataFrame()
                
            if not roster_data.empty:
                all_stats['roster'] = roster_data
                logger.info(f"Fetched {len(roster_data)} roster records")
        except Exception as e:
            logger.warning(f"Could not fetch roster data: {str(e)}")
        
        return all_stats
    
    def fetch_recent_weekly_stats(self, weeks_back: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Fetch recent weekly statistics for the most current data.
        
        Args:
            weeks_back: Number of recent weeks to fetch
            
        Returns:
            Dictionary of DataFrames containing recent weekly stats
        """
        logger.info(f"Fetching recent weekly statistics (last {weeks_back} weeks)...")
        
        current_year = datetime.now().year
        weekly_stats = {}
        
        try:
            # Try to get weekly data for current season
            weekly_data = nfl.import_weekly_data([current_year], s_type='REG')
            
            if not weekly_data.empty:
                # Get the most recent weeks
                max_week = weekly_data['week'].max()
                recent_weeks = list(range(max(1, max_week - weeks_back + 1), max_week + 1))
                
                recent_data = weekly_data[weekly_data['week'].isin(recent_weeks)]
                
                if not recent_data.empty:
                    logger.info(f"Fetched {len(recent_data)} recent weekly records from weeks {min(recent_weeks)}-{max(recent_weeks)}")
                    
                    # Split by stat types similar to seasonal data
                    for stat_type in ['passing', 'rushing', 'receiving']:
                        if stat_type == 'passing':
                            if 'attempts' in recent_data.columns:
                                stats_df = recent_data[recent_data['attempts'] > 0].copy()
                        elif stat_type == 'rushing':
                            if 'carries' in recent_data.columns:
                                stats_df = recent_data[recent_data['carries'] > 0].copy()
                        elif stat_type == 'receiving':
                            if 'receptions' in recent_data.columns:
                                stats_df = recent_data[recent_data['receptions'] > 0].copy()
                        
                        if not stats_df.empty:
                            weekly_stats[f'{stat_type}_weekly'] = stats_df
                            logger.info(f"Filtered {len(stats_df)} recent {stat_type} weekly records")
                else:
                    logger.warning("No recent weekly data found")
            else:
                logger.warning("No weekly data available for current season")
                
        except Exception as e:
            logger.warning(f"Could not fetch recent weekly data: {str(e)}")
        
        return weekly_stats
    
    def fetch_current_season_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch the most current season data available.
        
        Returns:
            Dictionary of DataFrames containing current season stats
        """
        current_year = datetime.now().year
        logger.info(f"Fetching current season ({current_year}) data...")
        
        current_stats = {}
        
        try:
            # Try current year first
            current_data = nfl.import_seasonal_data([current_year], s_type='REG')
            
            if current_data.empty:
                # If current year is empty, try previous year
                logger.info(f"No data for {current_year}, trying {current_year - 1}...")
                current_data = nfl.import_seasonal_data([current_year - 1], s_type='REG')
                current_year = current_year - 1
            
            if not current_data.empty:
                logger.info(f"Fetched {len(current_data)} current season ({current_year}) records")
                
                # Split by stat types
                for stat_type in ['passing', 'rushing', 'receiving']:
                    if stat_type == 'passing':
                        if 'attempts' in current_data.columns:
                            stats_df = current_data[current_data['attempts'] > 0].copy()
                    elif stat_type == 'rushing':
                        if 'carries' in current_data.columns:
                            stats_df = current_data[current_data['carries'] > 0].copy()
                    elif stat_type == 'receiving':
                        if 'receptions' in current_data.columns:
                            stats_df = current_data[current_data['receptions'] > 0].copy()
                    
                    if not stats_df.empty:
                        current_stats[f'{stat_type}_current'] = stats_df
                        logger.info(f"Filtered {len(stats_df)} current {stat_type} records")
            else:
                logger.warning("No current season data available")
                
        except Exception as e:
            logger.warning(f"Could not fetch current season data: {str(e)}")
        
        return current_stats
    
    def fetch_recent_weekly_stats(self, years: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch recent weekly NFL player statistics.
        
        Args:
            years: List of years to fetch (default: last 3 years)
            
        Returns:
            Dictionary of DataFrames containing weekly stats
        """
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 2, current_year + 1))  # Last 3 years
            
        logger.info(f"Fetching weekly NFL player stats for years: {min(years)}-{max(years)}")
        
        weekly_stats = {}
        
        try:
            # Fetch weekly data
            logger.info("Fetching weekly statistics...")
            weekly_data = nfl.import_weekly_data(years, s_type='REG')
            
            if not weekly_data.empty:
                logger.info(f"Fetched {len(weekly_data)} weekly records")
                logger.info(f"Weekly data columns: {list(weekly_data.columns)}")
                
                # Split into stat types similar to seasonal data
                stat_types = ['passing', 'rushing', 'receiving']
                
                for stat_type in stat_types:
                    if stat_type == 'passing':
                        if 'attempts' in weekly_data.columns:
                            stats_df = weekly_data[weekly_data['attempts'] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                    elif stat_type == 'rushing':
                        if 'carries' in weekly_data.columns:
                            stats_df = weekly_data[weekly_data['carries'] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                    elif stat_type == 'receiving':
                        if 'receptions' in weekly_data.columns:
                            stats_df = weekly_data[weekly_data['receptions'] > 0].copy()
                        else:
                            stats_df = pd.DataFrame()
                    
                    if not stats_df.empty:
                        weekly_stats[f'weekly_{stat_type}'] = stats_df
                        logger.info(f"Filtered {len(stats_df)} weekly {stat_type} records")
                    else:
                        logger.warning(f"No weekly data found for {stat_type}")
            else:
                logger.warning("No weekly data retrieved")
                
        except Exception as e:
            logger.warning(f"Could not fetch weekly data: {str(e)}")
        
        return weekly_stats
    
    def fetch_current_roster_data(self) -> pd.DataFrame:
        """
        Fetch the most current roster data available.
        
        Returns:
            DataFrame with current roster information
        """
        current_year = datetime.now().year
        roster_data = pd.DataFrame()
        
        # Try to get roster data for current year, fall back to previous years if needed
        for year in [current_year, current_year - 1, current_year - 2]:
            try:
                logger.info(f"Attempting to fetch roster data for {year}...")
                
                # Try different possible roster functions
                if hasattr(nfl, 'import_rosters'):
                    roster_data = nfl.import_rosters([year])
                elif hasattr(nfl, 'import_roster_data'):
                    roster_data = nfl.import_roster_data([year])
                elif hasattr(nfl, 'import_team_desc'):
                    # Sometimes roster info is in team descriptions
                    roster_data = nfl.import_team_desc()
                
                if not roster_data.empty:
                    logger.info(f"Successfully fetched roster data for {year}: {len(roster_data)} records")
                    break
                    
            except Exception as e:
                logger.warning(f"Could not fetch roster data for {year}: {str(e)}")
                continue
        
        return roster_data
    
    def save_raw_data(self, data_dict: Dict[str, pd.DataFrame], format: str = 'parquet') -> None:
        """
        Save raw data to local files.
        
        Args:
            data_dict: Dictionary of DataFrames to save
            format: File format ('csv' or 'parquet')
        """
        logger.info(f"Saving raw data in {format} format...")
        
        for data_type, df in data_dict.items():
            if format.lower() == 'csv':
                filepath = os.path.join(self.raw_data_path, f"{data_type}_stats.csv")
                df.to_csv(filepath, index=False)
            elif format.lower() == 'parquet':
                filepath = os.path.join(self.raw_data_path, f"{data_type}_stats.parquet")
                df.to_parquet(filepath, index=False)
            else:
                raise ValueError("Format must be 'csv' or 'parquet'")
                
            logger.info(f"Saved {data_type} data to {filepath}")
    
    def load_raw_data(self, format: str = 'parquet') -> Dict[str, pd.DataFrame]:
        """
        Load raw data from local files.
        
        Args:
            format: File format ('csv' or 'parquet')
            
        Returns:
            Dictionary of DataFrames
        """
        logger.info(f"Loading raw data from {format} files...")
        
        data_dict = {}
        file_extension = f".{format}"
        
        for filename in os.listdir(self.raw_data_path):
            if filename.endswith(file_extension):
                data_type = filename.replace(f"_stats{file_extension}", "")
                filepath = os.path.join(self.raw_data_path, filename)
                
                try:
                    if format.lower() == 'csv':
                        df = pd.read_csv(filepath)
                    elif format.lower() == 'parquet':
                        df = pd.read_parquet(filepath)
                    
                    data_dict[data_type] = df
                    logger.info(f"Loaded {data_type} data: {len(df)} records")
                    
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {str(e)}")
        
        return data_dict
    
    def clean_and_preprocess(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Clean and preprocess the NFL data.
        
        Args:
            data_dict: Dictionary of raw DataFrames
            
        Returns:
            Cleaned and preprocessed DataFrame
        """
        logger.info("Starting data cleaning and preprocessing...")
        
        # Combine all statistical data
        combined_stats = []
        
        for stat_type, df in data_dict.items():
            if stat_type == 'roster':
                continue  # Handle roster separately for metadata
                
            # Add stat type identifier
            df_copy = df.copy()
            df_copy['stat_type'] = stat_type
            combined_stats.append(df_copy)
        
        if not combined_stats:
            raise ValueError("No statistical data found to process")
        
        # Combine all stats
        df_combined = pd.concat(combined_stats, ignore_index=True, sort=False)
        
        # Add roster metadata if available
        if 'roster' in data_dict:
            roster_df = data_dict['roster']
            # Merge on player_id and season if columns exist
            merge_cols = []
            if 'player_id' in df_combined.columns and 'player_id' in roster_df.columns:
                merge_cols.append('player_id')
            if 'season' in df_combined.columns and 'season' in roster_df.columns:
                merge_cols.append('season')
            
            if merge_cols:
                df_combined = df_combined.merge(
                    roster_df[merge_cols + ['position', 'team', 'birth_date']].drop_duplicates(),
                    on=merge_cols,
                    how='left',
                    suffixes=('', '_roster')
                )
        
        logger.info(f"Combined dataset shape: {df_combined.shape}")
        
        # Clean the data
        df_cleaned = self._clean_data(df_combined)
        
        return df_cleaned
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the combined dataset.
        
        Args:
            df: Combined DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning data...")
        
        initial_rows = len(df)
        
        # Remove rows where player_name or player_id is missing
        if 'player_name' in df.columns:
            df = df.dropna(subset=['player_name'])
        elif 'player_id' in df.columns:
            df = df.dropna(subset=['player_id'])
        
        # Fill missing numerical values with 0 (common for NFL stats)
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        df[numerical_columns] = df[numerical_columns].fillna(0)
        
        # Fill missing categorical values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['player_name', 'player_id']:
                df[col] = df[col].fillna('Unknown')
        
        # Remove duplicate records
        duplicate_cols = ['player_id', 'season'] if 'player_id' in df.columns else ['player_name', 'season']
        if 'stat_type' in df.columns:
            duplicate_cols.append('stat_type')
        
        df = df.drop_duplicates(subset=duplicate_cols, keep='first')
        
        logger.info(f"Cleaning removed {initial_rows - len(df)} rows")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for machine learning.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        df_features = df.copy()
        
        # Calculate age if birth_date is available
        if 'birth_date' in df_features.columns and 'season' in df_features.columns:
            df_features['birth_date'] = pd.to_datetime(df_features['birth_date'], errors='coerce')
            df_features['age'] = df_features['season'] - df_features['birth_date'].dt.year
        
        # Calculate games-based metrics if games column exists
        games_col = None
        for col in ['games', 'g', 'gp']:
            if col in df_features.columns:
                games_col = col
                break
        
        if games_col:
            # Per-game statistics
            stat_columns = df_features.select_dtypes(include=[np.number]).columns
            for col in stat_columns:
                if col not in [games_col, 'season', 'age'] and df_features[col].sum() > 0:
                    # Only calculate per-game stats for players with games > 0
                    mask = df_features[games_col] > 0
                    df_features[f'{col}_per_game'] = 0.0
                    df_features.loc[mask, f'{col}_per_game'] = df_features.loc[mask, col] / df_features.loc[mask, games_col]
        
        # Add career experience (years in league)
        if 'player_id' in df_features.columns and 'season' in df_features.columns:
            player_first_season = df_features.groupby('player_id')['season'].min()
            df_features = df_features.merge(
                player_first_season.rename('first_season'),
                left_on='player_id',
                right_index=True,
                how='left'
            )
            df_features['years_experience'] = df_features['season'] - df_features['first_season']
        
        # Position groupings (if position exists)
        if 'position' in df_features.columns:
            df_features['position_group'] = df_features['position'].map(self._get_position_group)
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        
        return df_features
    
    def _get_position_group(self, position: str) -> str:
        """Map specific positions to broader position groups."""
        if pd.isna(position) or position == 'Unknown':
            return 'Unknown'
        
        position = position.upper()
        
        if position in ['QB']:
            return 'QB'
        elif position in ['RB', 'FB']:
            return 'RB'
        elif position in ['WR']:
            return 'WR'
        elif position in ['TE']:
            return 'TE'
        elif position in ['LT', 'LG', 'C', 'RG', 'RT', 'OL', 'T', 'G']:
            return 'OL'
        elif position in ['DE', 'DT', 'NT', 'DL']:
            return 'DL'
        elif position in ['LB', 'ILB', 'OLB', 'MLB']:
            return 'LB'
        elif position in ['CB', 'S', 'SS', 'FS', 'DB']:
            return 'DB'
        elif position in ['K', 'P', 'LS']:
            return 'ST'
        else:
            return 'Other'
    
    def normalize_and_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical features and encode categorical features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Processed DataFrame ready for ML
        """
        logger.info("Normalizing and encoding features...")
        
        df_processed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove identifier columns from processing
        id_cols = ['player_id', 'player_name', 'team_id']
        numerical_cols = [col for col in numerical_cols if col not in id_cols]
        categorical_cols = [col for col in categorical_cols if col not in id_cols]
        
        # Handle infinite and NaN values before normalization
        if numerical_cols:
            # Replace infinite values with NaN, then fill with 0
            df_processed[numerical_cols] = df_processed[numerical_cols].replace([np.inf, -np.inf], np.nan)
            df_processed[numerical_cols] = df_processed[numerical_cols].fillna(0)
            
            # Normalize numerical features
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            logger.info(f"Normalized {len(numerical_cols)} numerical features")
        
        # One-hot encode categorical features
        categorical_features = []
        for col in categorical_cols:
            if col in ['position', 'position_group', 'team', 'stat_type']:
                # Create dummy variables
                dummies = pd.get_dummies(df_processed[col], prefix=col, dummy_na=False)
                df_processed = pd.concat([df_processed, dummies], axis=1)
                categorical_features.extend(dummies.columns.tolist())
                
                # Drop original column
                df_processed = df_processed.drop(columns=[col])
        
        logger.info(f"Created {len(categorical_features)} categorical features")
        logger.info(f"Final processed dataset shape: {df_processed.shape}")
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_nfl_data.parquet') -> None:
        """
        Save processed data.
        
        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        filepath = os.path.join(self.processed_data_path, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")
        
        # Also save a sample CSV for inspection
        csv_path = filepath.replace('.parquet', '_sample.csv')
        df.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Saved sample CSV to {csv_path}")
    
    def save_interpretable_data(self, df: pd.DataFrame, filename: str = 'interpretable_nfl_data.parquet') -> None:
        """
        Save interpretable data without normalization.
        
        Args:
            df: DataFrame with engineered features but not normalized
            filename: Output filename
        """
        filepath = os.path.join(self.processed_data_path, filename)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved interpretable data to {filepath}")
        
        # Also save a sample CSV for inspection
        csv_path = filepath.replace('.parquet', '_sample.csv')
        df.head(1000).to_csv(csv_path, index=False)
        logger.info(f"Saved interpretable sample CSV to {csv_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary of the processed dataset.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'years_covered': [],
            'positions': [],
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        if 'season' in df.columns:
            summary['years_covered'] = sorted(df['season'].unique().tolist())
        
        # Find position columns
        position_cols = [col for col in df.columns if col.startswith('position_')]
        if position_cols:
            summary['positions'] = position_cols
        
        return summary


def main():
    """Main execution function."""
    logger.info("Starting NFL Data Pipeline")
    
    # Initialize pipeline
    pipeline = NFLDataPipeline()
    
    try:
        # Step 1: Fetch data
        logger.info("Step 1: Fetching NFL player statistics...")
        current_year = datetime.now().year
        current_month = datetime.now().month
        
        # Determine the most recent NFL season available
        # NFL season typically runs Sept-Feb, so data for current year is available after March
        if current_month >= 3:  # March or later
            max_year = current_year
        else:  # January-February
            max_year = current_year - 1
            
        years = list(range(1999, max_year + 1))
        stat_types = ['passing', 'rushing', 'receiving']
        logger.info(f"Fetching data from 1999 to {max_year} (most recent available season)")
        
        # Fetch seasonal data
        raw_data = pipeline.fetch_player_stats(years=years, stat_types=stat_types)
        
        # Fetch recent weekly data for more current information
        logger.info("Fetching recent weekly data...")
        weekly_data = pipeline.fetch_recent_weekly_stats()
        
        # Fetch current season data for most up-to-date stats
        logger.info("Fetching current season data...")
        current_data = pipeline.fetch_current_season_data()
        
        # Combine all data sources
        all_data = {**raw_data, **weekly_data, **current_data}
        
        # Fetch current roster data
        logger.info("Fetching current roster data...")
        current_roster = pipeline.fetch_current_roster_data()
        if not current_roster.empty:
            raw_data['current_roster'] = current_roster
        
        # Combine weekly data with seasonal data
        raw_data.update(weekly_data)
        
        # Step 2: Save raw data
        logger.info("Step 2: Saving raw data...")
        pipeline.save_raw_data(raw_data, format='parquet')
        
        # Step 3: Load and clean data
        logger.info("Step 3: Cleaning and preprocessing data...")
        cleaned_data = pipeline.clean_and_preprocess(raw_data)
        
        # Step 4: Feature engineering
        logger.info("Step 4: Engineering features...")
        featured_data = pipeline.engineer_features(cleaned_data)
        
        # Step 5: Save interpretable data (before normalization)
        logger.info("Step 5: Saving interpretable data...")
        pipeline.save_interpretable_data(featured_data)
        
        # Step 6: Normalize and encode
        logger.info("Step 6: Normalizing and encoding features...")
        processed_data = pipeline.normalize_and_encode(featured_data)
        
        # Step 7: Save processed data
        logger.info("Step 7: Saving processed data...")
        pipeline.save_processed_data(processed_data)
        
        # Step 8: Generate summary
        summary = pipeline.get_data_summary(featured_data)  # Use interpretable data for summary
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Data Summary: {summary}")
        
        return processed_data, summary
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    processed_data, summary = main()
    print(f"\nPipeline completed! Processed {summary['total_records']} records with {summary['total_features']} features.")