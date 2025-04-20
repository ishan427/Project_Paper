import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, Input
import os
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp
import time
from tensorflow.keras.optimizers import schedules

print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {pd.__version__}")
print(f"Run with GPU: {tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else tf.config.list_physical_devices('GPU')}")

# Step 1: Load or create food freshness dataset with sensor readings
real_data = pd.DataFrame([
    # Fresh food conditions
    [4.5, 60, 100, 85, 4.2, 4.1, 250, 6.8],
    [5.2, 58, 110, 95, 3.8, 4.3, 270, 6.9],
    [6.1, 62, 120, 105, 5.1, 5.0, 280, 6.7],
    [7.0, 63, 130, 115, 5.5, 5.2, 300, 6.6],
    [5.8, 61, 115, 100, 5.0, 4.8, 265, 6.7],
    [6.5, 59, 125, 110, 5.3, 5.1, 290, 6.8],
    [4.8, 57, 105, 90, 4.5, 4.4, 260, 6.9],
    [5.5, 60, 118, 98, 4.7, 4.6, 275, 6.8],
    
    # Spoiled food conditions
    [19.5, 85, 350, 280, 13.5, 18.8, 500, 5.3],
    [21.0, 90, 390, 310, 15.0, 19.5, 550, 5.0],
    [22.2, 88, 420, 325, 16.2, 20.1, 600, 4.8],
    [20.8, 91, 410, 330, 14.8, 19.7, 580, 4.9],
    [23.0, 92, 440, 340, 16.9, 21.5, 620, 4.7],
    [21.5, 89, 400, 315, 15.5, 20.0, 560, 5.1],
    [22.8, 93, 430, 335, 16.5, 21.0, 610, 4.8],
    [20.5, 87, 380, 300, 14.5, 19.0, 540, 5.2],

    # Additional fresh samples from the user data
    [4.8, 59, 105, 90, 4.5, 4.4, 260, 6.9],
], columns=['ambient_temp', 'humidity', 'gas', 'voc', 'co', 'core_temp', 'light', 'ph'])

# Define freshness label for each row based on temperature threshold
freshness = []
for i, row in real_data.iterrows():
    if row['ambient_temp'] < 10:  # Using temperature as the main indicator of freshness
        freshness.append('fresh')
    else:
        freshness.append('spoiled')
real_data['freshness'] = freshness

# Define data ranges for validation
sensor_ranges = {
    'ambient_temp': (0, 45),    # Celsius
    'humidity': (10, 100),      # Percentage
    'gas': (50, 1000),          # PPM
    'voc': (50, 800),           # PPB
    'co': (0, 50),              # PPM
    'core_temp': (0, 85),       # Celsius
    'light': (0, 1000),         # Lux
    'ph': (0, 14)               # pH scale
}

# Create freshness-specific ranges for more accurate generation
freshness_ranges = {
    'fresh': {
        'ambient_temp': (4, 8),
        'humidity': (55, 65),
        'gas': (95, 140),
        'voc': (80, 120),
        'co': (3.5, 6.0),
        'core_temp': (4.0, 5.5),
        'light': (245, 310),
        'ph': (6.5, 7.0)
    },
    'spoiled': {
        'ambient_temp': (19, 24),
        'humidity': (84, 94),
        'gas': (345, 445),
        'voc': (275, 345),
        'co': (13, 17),
        'core_temp': (18, 22),
        'light': (495, 625),
        'ph': (4.6, 5.4)
    }
}

# Data augmentation: Create additional synthetic samples using the existing data
def augment_data(data, num_augmented=50):
    """Create additional training samples through interpolation and small random variations"""
    original_data = data.drop('freshness', axis=1).values
    augmented_data = []
    
    for _ in range(num_augmented):
        # Select two random samples
        idx1, idx2 = np.random.choice(len(original_data), 2, replace=False)
        sample1, sample2 = original_data[idx1], original_data[idx2]
        
        # Interpolate between them with a random weight
        t = np.random.uniform(0.1, 0.9)
        interpolated = t * sample1 + (1-t) * sample2
        
        # Add small random noise (within 2% range of the original feature values)
        noise_scale = 0.02 * (sample1 + sample2) / 2
        noise = np.random.normal(0, noise_scale)
        augmented_sample = interpolated + noise
        
        # Clip values to ensure they're in valid ranges
        for i, (col, (min_val, max_val)) in enumerate(sensor_ranges.items()):
            augmented_sample[i] = np.clip(augmented_sample[i], min_val, max_val)
            
        augmented_data.append(augmented_sample)
    
    return np.array(augmented_data)

# Augment the real data to have more training samples
augmented_array = augment_data(real_data, num_augmented=100)
columns = real_data.drop('freshness', axis=1).columns
augmented_df = pd.DataFrame(augmented_array, columns=columns)

# Add freshness labels to augmented data
augmented_freshness = []
for i, row in augmented_df.iterrows():
    if row['ambient_temp'] < 10:
        augmented_freshness.append('fresh')
    else:
        augmented_freshness.append('spoiled')
augmented_df['freshness'] = augmented_freshness

# Combine original and augmented data
combined_data = pd.concat([real_data, augmented_df], ignore_index=True)

# Normalize data (only the sensor readings, not the freshness)
data_for_scaling = combined_data.drop('freshness', axis=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_for_scaling)
scaled_data = scaled_data.astype(np.float32)  # Ensure float32 for TensorFlow

# Prepare freshness labels as one-hot encoded values for conditional GAN
freshness_mapping = {'fresh': 0, 'spoiled': 1}
freshness_indices = np.array([freshness_mapping[c] for c in combined_data['freshness']])
freshness_onehot = tf.keras.utils.to_categorical(freshness_indices, num_classes=2)

# Store label information for later use
real_data_only_values = real_data.drop('freshness', axis=1).values
real_data_freshness = np.array([freshness_mapping[c] for c in real_data['freshness']])

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Model parameters
latent_dim = 64  # Increased from 32
data_dim = data_for_scaling.shape[1]
num_conditions = 2  # fresh, spoiled
condition_dim = num_conditions

# Define learning rate schedule for better convergence
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=0.0002,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)

# Build simplified generator without problematic attention mechanism
def build_generator(latent_dim, data_dim, condition_dim):
    # Latent space input
    z_input = Input(shape=(latent_dim,))
    
    # Condition input
    condition_input = Input(shape=(condition_dim,))
    
    # Concatenate latent vector and condition
    z_condition = tf.keras.layers.Concatenate()([z_input, condition_input])
    
    # First dense layer
    x = Dense(128)(z_condition)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Enhanced feature extraction with residual connections
    residual_block = Dense(128)(x)
    residual_block = LeakyReLU(0.2)(residual_block)
    residual_block = BatchNormalization()(residual_block)
    residual_block = Dense(128)(residual_block)
    residual_block = LeakyReLU(0.2)(residual_block)
    residual_block = BatchNormalization()(residual_block)
    x = tf.keras.layers.Add()([x, residual_block])
    
    # Deep layers
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    # Additional layer with residual connection
    residual = x
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])  # Residual connection
    
    # Output layer
    output = Dense(data_dim, activation='sigmoid')(x)
    
    # Define model
    model = Model([z_input, condition_input], output, name="Generator")
    return model

# Build enhanced discriminator with feature matching
def build_discriminator(data_dim, condition_dim):
    # Real/fake data input
    data_input = Input(shape=(data_dim,))
    
    # Condition input
    condition_input = Input(shape=(condition_dim,))
    
    # Concatenate data and condition
    x = tf.keras.layers.Concatenate()([data_input, condition_input])
    
    # First dense layer
    x = Dense(128)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # Hidden layers
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # Additional hidden layer
    x = Dense(256)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)
    
    # Extract features for feature matching loss
    features = Dense(128, name="features")(x)
    features = LeakyReLU(0.2)(features)
    
    # Output layer for real/fake classification
    validity = Dense(1, activation='sigmoid')(features)
    
    # Define model with both validity output and feature output
    model = Model([data_input, condition_input], [validity, features], name="Discriminator")
    return model

# Create models
generator = build_generator(latent_dim, data_dim, condition_dim)
discriminator = build_discriminator(data_dim, condition_dim)

# Define optimizers with learning rate schedule for Keras 3 compatibility
g_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5)

# Define loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
mse = tf.keras.losses.MeanSquaredError()

# WGAN-GP gradient penalty function
def gradient_penalty(discriminator, batch_size, real_samples, fake_samples, conditions):
    """Calculates the gradient penalty for WGAN-GP"""
    # Generate random interpolation factors
    alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
    
    # Create interpolated samples
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # Get discriminator output for interpolated samples
        pred, _ = discriminator([interpolated, conditions], training=True)
    
    # Calculate gradients with respect to input
    grads = gp_tape.gradient(pred, interpolated)
    # Calculate gradient norm
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
    # Calculate gradient penalty
    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp

# Train step function using GradientTape and feature matching
@tf.function
def train_step(real_samples, conditions, batch_size):
    # Sample from latent space
    noise = tf.random.normal([batch_size, latent_dim])
    
    # Label smoothing for more stable training
    real_labels = tf.ones((batch_size, 1)) * 0.9  # 0.9 instead of 1.0
    fake_labels = tf.zeros((batch_size, 1)) + 0.1  # 0.1 instead of 0.0
    
    # Train discriminator
    with tf.GradientTape() as d_tape:
        # Generate fake samples
        fake_samples = generator([noise, conditions], training=True)
        
        # Get discriminator outputs
        real_validity, real_features = discriminator([real_samples, conditions], training=True)
        fake_validity, fake_features = discriminator([fake_samples, conditions], training=True)
        
        # Calculate discriminator losses
        d_loss_real = cross_entropy(real_labels, real_validity)
        d_loss_fake = cross_entropy(fake_labels, fake_validity)
        d_loss_main = d_loss_real + d_loss_fake
        
        # Add gradient penalty for WGAN-GP style training (improves stability)
        gp = gradient_penalty(discriminator, batch_size, real_samples, fake_samples, conditions) 
        d_loss = d_loss_main + 10.0 * gp  # Lambda = 10 is typical for GP
    
    # Calculate gradients and update discriminator
    d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    
    # Calculate accuracy for monitoring
    real_acc = tf.reduce_mean(tf.cast(real_validity > 0.5, tf.float32))
    fake_acc = tf.reduce_mean(tf.cast(fake_validity < 0.5, tf.float32))
    d_acc = (real_acc + fake_acc) / 2.0
    
    # Train generator
    with tf.GradientTape() as g_tape:    
        # Generate new fake samples
        fake_samples = generator([noise, conditions], training=True)
        
        # Get discriminator prediction on fake samples
        fake_validity, fake_features = discriminator([fake_samples, conditions], training=True)    
        
        # Feature matching: match discriminator features for real and fake
        _, real_features = discriminator([real_samples, conditions], training=False)
        
        # Calculate generator losses
        g_loss_adv = cross_entropy(real_labels, fake_validity)  # Adversarial loss
        g_loss_feat = mse(real_features, fake_features)  # Feature matching loss
        
        # Combined loss with weighting
        g_loss = g_loss_adv + 10.0 * g_loss_feat  # Feature matching has higher weight
    
    # Calculate gradients and update generator
    g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
    
    return d_loss, d_acc, g_loss, g_loss_adv, g_loss_feat

# Function to generate samples with condition control
def generate_samples(n_samples, condition=None):
    """Generate synthetic samples with optional condition control"""
    # Create noise vector
    noise = tf.random.normal([n_samples, latent_dim])
    
    # Set condition if specified, otherwise sample randomly        
    if condition is not None:
        # Create one-hot encoded condition
        if condition == 'fresh':
            condition_vec = np.tile([1, 0], (n_samples, 1))
        elif condition == 'spoiled':
            condition_vec = np.tile([0, 1], (n_samples, 1))
        else:
            raise ValueError(f"Unknown condition: {condition}")
    else:
        # Random conditions
        condition_indices = np.random.randint(0, 2, size=n_samples)
        condition_vec = tf.keras.utils.to_categorical(condition_indices, num_classes=2)
    
    # Generate samples
    condition_tensor = tf.convert_to_tensor(condition_vec, dtype=tf.float32)
    generated_samples = generator([noise, condition_tensor], training=False)
    
    return generated_samples.numpy(), condition_vec

# Create output directory
output_dir = "gan_output"
os.makedirs(output_dir, exist_ok=True)

# Training parameters
epochs = 20000  # Increased from 15000
batch_size = 16 if len(scaled_data) > 32 else 8  # Adapt to data size
history = {'d_loss': [], 'd_acc': [], 'g_loss': [], 'g_adv_loss': [], 'g_feat_loss': []}

# Convert data to tensors
real_samples_tensor = tf.convert_to_tensor(scaled_data, dtype=tf.float32)
condition_tensor = tf.convert_to_tensor(freshness_onehot, dtype=tf.float32)

# Early stopping parameters
patience = 2000  # Increased from 1000 for more training opportunity
best_g_loss = float('inf')
patience_counter = 0

# Run training loop
print("\nStarting enhanced conditional GAN training for food freshness detection...")
try:
    start_time = time.time()
    for epoch in range(epochs):
        # Get batch of real samples with their conditions
        idx = np.random.randint(0, len(scaled_data), batch_size)
        real_batch = tf.gather(real_samples_tensor, idx)
        condition_batch = tf.gather(condition_tensor, idx)
        
        # Train on batch
        d_loss, d_acc, g_loss, g_adv_loss, g_feat_loss = train_step(real_batch, condition_batch, batch_size)
        
        # Store history
        history['d_loss'].append(float(d_loss))
        history['d_acc'].append(float(d_acc))
        history['g_loss'].append(float(g_loss))
        history['g_adv_loss'].append(float(g_adv_loss))
        history['g_feat_loss'].append(float(g_feat_loss))
        
        # Early stopping check (using combined g_loss)
        if float(g_loss) < best_g_loss:
            best_g_loss = float(g_loss)
            patience_counter = 0
            
            # Save best model
            if epoch > 1000:
                generator.save(f"{output_dir}/generator_best.h5")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Display progress
        if epoch % 500 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}: [D loss: {float(d_loss):.4f}, acc: {float(d_acc)*100:.2f}%] " +
                  f"[G loss: {float(g_loss):.4f}, adv: {float(g_adv_loss):.4f}, feat: {float(g_feat_loss):.4f}] " +
                  f"Time: {elapsed:.1f}s")
            if epoch % 2000 == 0 and epoch > 0:
                # Save models
                generator.save(f"{output_dir}/generator_{epoch}.h5")
                
                # Generate sample data for each freshness category for inspection
                for condition in ['fresh', 'spoiled']:
                    samples, _ = generate_samples(3, condition)
                    samples_rescaled = scaler.inverse_transform(samples)
                    
                    print(f"\nSample {condition} food generated data:")
                    sample_df = pd.DataFrame(samples_rescaled, columns=data_for_scaling.columns)
                    print(sample_df.round(2))

    print(f"\nTraining completed in {time.time() - start_time:.1f} seconds!")
    
    # Try to load best model if it exists, otherwise use current model
    try:
        if os.path.exists(f"{output_dir}/generator_best.h5"):
            generator = tf.keras.models.load_model(f"{output_dir}/generator_best.h5")
            print("Loaded best generator model for final generation")            
    except Exception as e:
        print(f"Could not load best model: {e}. Using current model instead.")
    
except Exception as e:
    print(f"Error during training: {str(e)}")
    
# Step 4: Generate and validate synthetic samples with improved filtering
print("\nGenerating synthetic samples for each food freshness category...")

# Generate samples for each freshness category with enhanced post-processing
def generate_freshness_specific_data(freshness, target_samples=200):
    """Generate high-quality samples for a specific freshness category with additional filtering"""
    print(f"Generating {freshness} food samples...")
    
    # Map freshness to one-hot
    if freshness == 'fresh':
        condition_idx = 0
        condition_vec = np.tile([1, 0], (target_samples*3, 1))  # Generate 3x samples to filter
    elif freshness == 'spoiled':
        condition_idx = 1
        condition_vec = np.tile([0, 1], (target_samples*3, 1))
    
    # Get freshness-specific ranges
    ranges = freshness_ranges[freshness]
    
    # Generate more samples than needed to allow for filtering
    noise = tf.random.normal([target_samples*3, latent_dim])
    condition_tensor = tf.convert_to_tensor(condition_vec, dtype=tf.float32)
    base_synthetic = generator([noise, condition_tensor], training=False).numpy()
    synthetic_rescaled = scaler.inverse_transform(base_synthetic)
    synthetic_df = pd.DataFrame(synthetic_rescaled, columns=data_for_scaling.columns)
    
    # Apply freshness-specific filtering
    filtered_df = synthetic_df.copy()
    
    # Filter based on freshness ranges first
    for feature, (min_val, max_val) in ranges.items():
        filtered_df = filtered_df[(filtered_df[feature] >= min_val) & (filtered_df[feature] <= max_val)]
    
    # If we don't have enough samples after strict filtering, relax the constraints a bit
    if len(filtered_df) < target_samples:
        filtered_df = synthetic_df.copy()
        
        # Look at real samples for this freshness category
        real_freshness_data = real_data[real_data['freshness'] == freshness].drop('freshness', axis=1)
        
        # Calculate feature means for this freshness
        feature_means = real_freshness_data.mean()
        
        # Calculate feature standard deviations for this freshness
        feature_stds = real_freshness_data.std()
        
        # Filter based on statistical distance from the mean (within 2.5 standard deviations)
        for feature in data_for_scaling.columns:
            mean_val = feature_means[feature]
            std_val = max(feature_stds[feature], 0.001)  # Avoid division by zero
            
            lower_bound = mean_val - 2.5 * std_val
            upper_bound = mean_val + 2.5 * std_val
            
            filtered_df = filtered_df[(filtered_df[feature] >= lower_bound) & 
                                    (filtered_df[feature] <= upper_bound)]
    
    # Clip values to realistic ranges
    for column, (min_val, max_val) in sensor_ranges.items():
        filtered_df[column] = filtered_df[column].clip(min_val, max_val)
    
    # Take only the requested number of samples if we have enough
    if len(filtered_df) > target_samples:
        filtered_df = filtered_df.iloc[:target_samples]
    
    print(f"Generated {len(filtered_df)} valid {freshness} food samples out of {target_samples} requested")
    return filtered_df

# Generate high-quality samples for each freshness category
fresh_data = generate_freshness_specific_data('fresh', 200)
spoiled_data = generate_freshness_specific_data('spoiled', 200)

# Generate general-purpose samples
n_samples = 5000
synthetic_data, _ = generate_samples(n_samples)
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=data_for_scaling.columns)

# Validate general samples against defined ranges
validated_data = synthetic_df.copy()
for column, (min_val, max_val) in sensor_ranges.items():
    # Clip values to realistic ranges
    validated_data[column] = validated_data[column].clip(min_val, max_val)
    
    # Count how many samples required correction
    out_of_range = ((synthetic_df[column] < min_val) | (synthetic_df[column] > max_val)).sum()
    if out_of_range > 0:
        print(f"Warning: {out_of_range} samples ({out_of_range/n_samples*100:.1f}%) for {column} were outside realistic range")

# Step 5: Visualize training history with additional metrics
def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['d_loss'], label='Discriminator')
    plt.plot(history['g_loss'], label='Generator Combined')
    plt.title('GAN Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot discriminator accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['d_acc'])
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    # Plot generator component losses
    plt.subplot(2, 2, 3)
    plt.plot(history['g_adv_loss'], label='Adversarial')
    plt.plot(history['g_feat_loss'], label='Feature Matching')
    plt.title('Generator Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot moving averages for smoother visualization
    window = min(50, len(history['d_loss'])//10)
    if window > 1:
        plt.subplot(2, 2, 4)
        g_loss_smooth = np.convolve(history['g_loss'], np.ones(window)/window, mode='valid')
        d_loss_smooth = np.convolve(history['d_loss'], np.ones(window)/window, mode='valid')
        plt.plot(g_loss_smooth, label='Generator')
        plt.plot(d_loss_smooth, label='Discriminator')
        plt.title(f'Smoothed Losses (Window={window})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_training_history.png")
    print(f"Enhanced training history plot saved to {output_dir}/enhanced_training_history.png")

# Step 6: Analyze quality of generated data with more detailed metrics
def evaluate_synthetic_data(real_df, synthetic_df):
    """Evaluate the quality of synthetic data against real data with detailed metrics"""
    print("\n=== Statistical Validation of Synthetic Data ===")
    
    # Remove condition column for analysis if it exists
    if 'freshness' in real_df.columns:
        real_df = real_df.drop('freshness', axis=1)
    
    # Statistical comparison
    stats_real = real_df.describe()
    stats_synthetic = synthetic_df.describe()
    
    print("\nReal Data Statistics:")
    print(stats_real.round(2))
    print("\nSynthetic Data Statistics:")
    print(stats_synthetic.round(2))
    
    # Calculate percentage differences in key statistics
    print("\nPercentage difference in statistics (synthetic vs. real):")
    mean_diff_pct = abs((stats_synthetic.loc['mean'] - stats_real.loc['mean']) / stats_real.loc['mean'] * 100)
    std_diff_pct = abs((stats_synthetic.loc['std'] - stats_real.loc['std']) / stats_real.loc['std'] * 100)
    print("Mean difference %:")
    print(mean_diff_pct.round(2))
    print("Std difference %:")
    print(std_diff_pct.round(2))
    
    # Calculate average statistical similarity score (lower is better)
    mean_similarity = mean_diff_pct.mean()
    std_similarity = std_diff_pct.mean()
    overall_similarity = (mean_similarity + std_similarity) / 2
    
    print(f"\nOverall statistical similarity score: {overall_similarity:.2f}% difference (lower is better)")
    
    # KS test for distribution similarity
    print("\nKolmogorov-Smirnov Test Results (p-values):")
    similar_count = 0
    for col in real_df.columns:
        stat, p_value = ks_2samp(real_df[col], synthetic_df[col])
        similarity = "Similar" if p_value > 0.05 else "Different"
        if similarity == "Similar":
            similar_count += 1
        print(f"{col}: p-value={p_value:.4f} ({similarity})")
    print(f"\nDistribution similarity: {similar_count}/{len(real_df.columns)} features statistically similar")
    
    # Plot histograms to compare distributions
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(real_df.columns):
        plt.subplot(3, 3, i+1)
        sns.histplot(real_df[col], color='blue', alpha=0.5, label='Real', kde=True)
        sns.histplot(synthetic_df[col], color='red', alpha=0.5, label='Synthetic', kde=True)
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_distribution_comparison.png")
    print(f"Distribution comparison plot saved to {output_dir}/enhanced_distribution_comparison.png")
    
    # Enhanced distribution visualization with KDE plots
    print("\nGenerating enhanced distribution visualization...")
    try:
        plt.figure(figsize=(16, 12))
        
        # Plot distributions separately for fresh and spoiled categories
        features = ['ambient_temp', 'humidity', 'gas', 'voc', 'co', 'core_temp', 'light', 'ph']
        for i, feature in enumerate(features):
            plt.subplot(4, 2, i+1)
            
            # Get fresh & spoiled data
            real_fresh = real_df[real_data['freshness'] == 'fresh'][feature]
            real_spoiled = real_df[real_data['freshness'] == 'spoiled'][feature]
            
            # Plot with better visualization
            sns.kdeplot(real_fresh, color='blue', label='Real Fresh', linestyle='-')
            sns.kdeplot(real_spoiled, color='red', label='Real Spoiled', linestyle='-')
            
            plt.title(f'Distribution of {feature}')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distributions_by_freshness.png")
        print(f"Enhanced distribution plots saved to {output_dir}/feature_distributions_by_freshness.png")
        
        # Continue with t-SNE visualization
        combined_data = np.vstack([real_df.values, synthetic_df.values[:len(real_df)*5]])
        combined_scaled = MinMaxScaler().fit_transform(combined_data)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_scaled)-1))
        tsne_results = tsne.fit_transform(combined_scaled)
        
        plt.figure(figsize=(12, 10))
        scatter1 = plt.scatter(tsne_results[:len(real_df), 0], tsne_results[:len(real_df), 1], 
                  c='blue', label='Real', alpha=0.7, s=70)
        scatter2 = plt.scatter(tsne_results[len(real_df):, 0], tsne_results[len(real_df):, 1], 
                  c='red', alpha=0.5, label='Synthetic', s=30)
        plt.title('t-SNE: Real vs Synthetic Data', fontsize=14)
        plt.legend(handles=[scatter1, scatter2], fontsize=12)
        plt.savefig(f"{output_dir}/enhanced_tsne_comparison.png")
        print(f"Enhanced t-SNE plot saved to {output_dir}/enhanced_tsne_comparison.png")
    except Exception as e:
        print(f"Error generating t-SNE: {str(e)}")
    
    return stats_real, stats_synthetic, overall_similarity

# Create fruit spoilage dataset
def create_fruit_spoilage_dataset():
    """Create a biologically accurate fruit spoilage dataset with 2000 samples based on scientific principles."""
    import random
    from datetime import datetime, timedelta
    from scipy import stats
    
    print("\nGenerating scientifically accurate fruit spoilage dataset with 2000 entries...")
    
    # Define fruit categories with specific biological properties and spoilage patterns
    # Format: [name, climacteric(T/F), respiration_rate(mg CO2/kg·h), ethylene_production(μL/kg·h), 
    #          sugar_content(%), spoilage_resistance(1-10), shelf_life_days, moisture_content(%)]
    fruit_categories = [
        # Climacteric fruits with enhanced spoilage characteristics
        ["Apple", True, (8, 25), (0.1, 1.5), (10, 14), 7, (14, 30), (84, 86)],
        ["Banana", True, (15, 60), (0.2, 280), (12, 20), 4, (7, 14), (74, 76)],
        ["Mango", True, (15, 70), (0.1, 4.0), (14, 18), 5, (7, 14), (82, 84)],
        ["Papaya", True, (15, 35), (0.2, 8.0), (6, 13), 3, (5, 10), (86, 89)],
        ["Avocado", True, (25, 80), (20, 100), (0.8, 1.5), 6, (7, 14), (72, 74)],
        ["Peach", True, (5, 70), (0.1, 60), (8, 12), 3, (3, 7), (87, 89)],
        ["Pear", True, (10, 30), (0.1, 10), (9, 16), 5, (5, 14), (83, 85)],
        ["Kiwi", True, (15, 30), (0.1, 1.0), (9, 14), 4, (14, 28), (82, 84)],
        ["Tomato", True, (10, 25), (0.5, 10), (3, 5), 5, (7, 14), (93, 95)],
        
        # Non-climacteric fruits with enhanced spoilage characteristics
        ["Strawberry", False, (10, 25), (0.01, 0.1), (5, 9), 2, (3, 7), (89, 91)],
        ["Orange", False, (5, 15), (0.01, 0.1), (8, 12), 6, (14, 30), (86, 88)],
        ["Pineapple", False, (10, 25), (0.05, 0.3), (12, 16), 6, (10, 21), (85, 87)],
        ["Cherry", False, (5, 20), (0.01, 0.1), (12, 18), 3, (3, 10), (80, 83)],
        ["Grape", False, (5, 10), (0.01, 0.1), (15, 25), 4, (7, 14), (80, 82)],
        ["Watermelon", False, (3, 10), (0.005, 0.05), (6, 10), 5, (14, 21), (91, 93)],
        ["Blueberry", False, (10, 20), (0.01, 0.1), (10, 15), 4, (10, 14), (83, 85)],
        ["Cucumber", False, (5, 15), (0.05, 0.2), (1, 3), 5, (10, 14), (95, 97)],
        ["Bell Pepper", False, (8, 18), (0.01, 0.1), (4, 6), 6, (14, 21), (92, 94)]
    ]
    
    # Define common spoilage microorganisms with their growth characteristics and specific effects
    # [name, optimal_temp_range, optimal_humidity_range, optimal_pH_range, key_effects_dict]
    spoilage_microorganisms = [
        ["Aspergillus niger", (25, 35), (80, 95), (3.0, 6.0), 
         {"gas": 1.5, "voc": 2.0, "ph": -0.8, "primary_targets": ["Grapes", "Strawberry", "Orange"]}],
        ["Botrytis cinerea", (15, 25), (85, 95), (3.0, 7.0), 
         {"gas": 1.2, "voc": 2.5, "ph": -0.5, "primary_targets": ["Strawberry", "Grape", "Kiwi", "Cherry"]}],
        ["Penicillium expansum", (20, 28), (80, 90), (3.5, 5.5), 
         {"gas": 1.8, "voc": 1.7, "ph": -1.0, "primary_targets": ["Apple", "Pear", "Cherry"]}],
        ["Rhizopus stolonifer", (25, 30), (85, 100), (4.5, 6.5), 
         {"gas": 2.0, "voc": 1.8, "ph": -0.7, "primary_targets": ["Peach", "Strawberry", "Tomato"]}],
        ["Colletotrichum gloeosporioides", (25, 30), (80, 95), (5.0, 7.0),
         {"gas": 1.3, "voc": 1.5, "ph": -0.6, "primary_targets": ["Mango", "Avocado", "Banana"]}],
        ["Erwinia carotovora", (25, 30), (85, 95), (6.0, 7.0),
         {"gas": 2.2, "voc": 2.0, "ph": -0.9, "primary_targets": ["Bell Pepper", "Cucumber", "Tomato"]}],
        ["Lactobacillus spp.", (30, 40), (70, 90), (4.0, 6.0),
         {"gas": 1.6, "voc": 1.3, "ph": -1.2, "primary_targets": ["Tomato", "Cucumber", "Bell Pepper"]}]
    ]
    
    # Define environmental storage scenarios with more detailed conditions
    # [name, temp_range, humidity_range, light_range, description, CO2_levels, O2_levels]
    storage_scenarios = [
        ["Optimal refrigeration", (2, 8), (85, 95), (0, 50), 
         "Proper cold chain storage", (300, 800), (20, 21)],
        ["Room temperature", (18, 25), (50, 70), (100, 300), 
         "Regular room conditions", (800, 1500), (19, 21)],
        ["Warm storage", (25, 35), (60, 80), (150, 400), 
         "Warm indoor conditions", (1000, 2000), (18, 20)],
        ["High humidity", (20, 30), (80, 95), (100, 300), 
         "Humid storage conditions", (800, 1800), (18, 20)],
        ["Transport condition", (15, 30), (60, 90), (0, 200), 
         "Variable transport conditions", (800, 2500), (17, 21)],
        ["Ethylene exposure", (10, 25), (60, 85), (50, 200),
         "Exposure to ethylene gas", (800, 1500), (18, 21)]
    ]

    # Enhanced biological spoilage progression functions
    def calculate_natural_gas_emission(fruit_type, days, temp, is_climacteric, spoilage_resistance, shelf_life):
        """Calculate gas emission based on fruit biology, time, and temperature with improved spoilage modeling"""
        # Base rate depends on fruit resistance to spoilage (lower resistance = higher base rate)
        base_rate = random.uniform(80, 110) * (1 + (10-spoilage_resistance)/20)
        
        # Calculate percentage of shelf life elapsed
        shelf_life_percent = min(1.0, days / shelf_life)
        
        # Climacteric fruits have a respiration and ethylene peak during ripening
        if is_climacteric:
            # Enhanced sigmoid curve with shelf life factored in
            ripening_factor = 1 + 5 * (1 / (1 + np.exp(-4 * (shelf_life_percent - 0.5))))
            # Temperature accelerates ripening (Q10 effect)
            temp_factor = 1 + 0.15 * (temp - 5)
        else:
            # More realistic non-climacteric progression
            ripening_factor = 1 + 0.8 * shelf_life_percent
            temp_factor = 1 + 0.08 * (temp - 5)
        
        # Exponential increase as food approaches spoilage
        if shelf_life_percent > 0.8:
            spoilage_acceleration = np.exp(2 * (shelf_life_percent - 0.8))
        else:
            spoilage_acceleration = 1.0
            
        # Calculate final gas emission
        emission = base_rate * ripening_factor * temp_factor * spoilage_acceleration
        
        # Add realistic variation
        variation = random.uniform(0.9, 1.1)
        
        return max(base_rate, emission * variation)
    
    def calculate_microbial_growth(days, temp, humidity, pH, shelf_life, spoilage_resistance, microorganism=None):
        """Enhanced microbial growth model based on environmental conditions and fruit resistance"""
        # Base growth rate
        if microorganism:
            # Get organism's preferred conditions
            opt_temp_range, opt_humidity_range, opt_pH_range, _ = microorganism[1:]
            
            # Calculate how optimal the conditions are (0-1 scale)
            temp_optimality = gaussian_optimality(temp, *opt_temp_range)
            humidity_optimality = gaussian_optimality(humidity, *opt_humidity_range)
            pH_optimality = gaussian_optimality(pH, *opt_pH_range)
            
            # Combined optimality score
            optimality = (temp_optimality + humidity_optimality + pH_optimality) / 3
        else:
            # General case
            # Temperature factors - most spoilage microbes grow faster at higher temps
            if temp < 5:  # Very slow growth at refrigeration temps
                temp_factor = 0.1
            else:
                # Growth accelerates between 5-35°C
                temp_factor = 0.2 * (temp - 5) if temp < 30 else 4.5 - 0.15 * (temp - 30)
                
            humidity_factor = 0.5 + 0.005 * humidity
            
            # Most spoilage organisms prefer slightly acidic to neutral pH
            if pH < 4.0:
                pH_factor = 0.4
            elif pH > 7.5:
                pH_factor = 0.6
            else:
                pH_factor = 1.0
            
            optimality = (temp_factor * humidity_factor * pH_factor) / 3
        
        # Shelf life percentage
        shelf_life_percent = min(1.0, days / shelf_life)
        
        # Fruit resistance factor (1-10 scale, higher = more resistant)
        resistance_factor = 1 - (spoilage_resistance / 15)  # Convert to 0-1 scale, inverted
        
        # Logistic growth curve over time with environmental modulation
        lag_phase = 0.3 * shelf_life  # Initial slow growth
        if days < lag_phase:
            time_factor = 0.1 + 0.9 * (days / lag_phase)
        else:
            # Exponential growth after lag phase
            time_factor = 1.0 + 2.0 * ((days - lag_phase) / (shelf_life - lag_phase))
        
        # Combine all factors for final growth
        growth = 20 * time_factor * optimality * (1 + resistance_factor)
        
        # Add stochastic variation
        variation = random.uniform(0.85, 1.15)
        
        return max(0, min(growth * variation, 350))  # Cap the growth
    
    def calculate_voc_production(days, temp, base_rate, is_climacteric, shelf_life):
        """Calculate volatile organic compounds with improved spoilage progression"""
        # Shelf life percentage
        shelf_life_percent = min(1.0, days / shelf_life)
        
        if is_climacteric:
            # VOCs increase significantly during ripening for climacteric fruits
            if shelf_life_percent < 0.6:
                # Normal ripening phase
                ripening_factor = 0.8 + 2.5 * shelf_life_percent
            else:
                # Over-ripening and spoilage phase - accelerated VOC production
                ripening_factor = 2.3 + 4 * (shelf_life_percent - 0.6)
        else:
            # More linear increase for non-climacteric, but still accelerates with spoilage
            if shelf_life_percent < 0.7:
                ripening_factor = 1 + 0.6 * shelf_life_percent
            else:
                ripening_factor = 1.42 + 3 * (shelf_life_percent - 0.7)
            
        # Temperature effect follows Q10 principle
        temp_factor = 1 + 0.1 * (temp - 5) if temp > 5 else 0.5
        
        # Calculate VOC with stochastic variation
        variation = random.uniform(0.9, 1.1)
        return base_rate * ripening_factor * temp_factor * variation
    
    def gaussian_optimality(value, min_opt, max_opt):
        """Calculate how optimal a value is using a Gaussian function (0-1 scale)"""
        mid_point = (min_opt + max_opt) / 2
        width = (max_opt - min_opt) / 2
        
        if width == 0:
            return 1.0 if value == mid_point else 0.0
            
        # Gaussian function centered at midpoint
        return np.exp(-0.5 * ((value - mid_point) / width) ** 2)
    
    # Function to generate accurate fruit samples based on scientific principles
    def generate_fruit_sample(sr_no, fruit_info, days_stored, storage_condition, forced_state=None):
        """Generate scientifically accurate fruit sample with enhanced spoilage modeling"""
        # Unpack fruit information with expanded parameters
        fruit_name, is_climacteric, respiration_range, ethylene_range, sugar_range, \
            spoilage_resistance, shelf_life_range, moisture_range = fruit_info
        
        # Determine shelf life for this specific sample
        shelf_life = random.uniform(*shelf_life_range)
        
        # Unpack storage information with enhanced parameters
        storage_name, temp_range, humidity_range, light_range, _, co2_range, o2_range = storage_condition
        
        # Generate base environmental conditions from storage scenario
        base_temp = random.uniform(*temp_range)
        base_humidity = random.uniform(*humidity_range)
        light = random.uniform(*light_range)
        co2_level = random.uniform(*co2_range)
        o2_level = random.uniform(*o2_range)
        
        # Add small daily fluctuations to simulate real conditions
        temp = round(base_temp + random.uniform(-1.0, 1.0), 1)
        humidity = round(base_humidity + random.uniform(-3.0, 3.0), 1)
        
        # Calculate respiration rate based on fruit type and conditions
        respiration_base = random.uniform(*respiration_range)
        # Temperature affects respiration (Q10 effect ~2-3 for most fruits)
        respiration_temp_factor = 2.5 ** ((temp - 10) / 10) if temp > 0 else 0.1
        respiration_rate = respiration_base * respiration_temp_factor
        
        # Calculate ethylene production
        ethylene_base = random.uniform(*ethylene_range)
        ethylene_temp_factor = 1.8 ** ((temp - 10) / 10) if temp > 0 else 0.05
        ethylene_production = ethylene_base * ethylene_temp_factor
        
        # Enhanced calculation for climacteric fruits - ethylene spike during ripening
        if is_climacteric and days_stored > 0.5 * shelf_life and days_stored < 0.8 * shelf_life:
            # Ethylene climacteric peak
            peak_factor = 3.0 * np.exp(-10 * ((days_stored/shelf_life - 0.65) ** 2))
            ethylene_production *= (1 + peak_factor)
        
        # Generate sugar content (decreases slightly during respiration)
        sugar_base = random.uniform(*sugar_range)
        sugar_percent = 1.0 - min(1.0, 0.005 * days_stored * respiration_temp_factor * (shelf_life/days_stored if days_stored > 0 else 1))
        sugar_content = sugar_base * sugar_percent
        
        # Generate moisture content
        moisture_content = random.uniform(*moisture_range)
        # Moisture decreases over time due to transpiration
        moisture_loss = min(0.15, 0.005 * days_stored * (temp/10))
        moisture_content *= (1 - moisture_loss)
        
        # Core temp is typically slightly lower than ambient in fresh produce
        core_temp = round(temp - random.uniform(0.2, 1.0), 1)
        
        # pH modeling - pH typically drops during spoilage
        if days_stored < 0.5 * shelf_life:
            # Fresh stage
            base_pH = random.uniform(6.0, 7.0)
            pH_change = -0.02 * days_stored
        else:
            # Ripening/spoilage stage - pH drops faster
            base_pH = random.uniform(5.0, 6.2)
            pH_change = -0.06 * (days_stored - 0.5 * shelf_life)
            
        pH = round(max(3.8, base_pH + pH_change), 1)
        
        # Determine whether microbial spoilage is present
        microbial_probability = min(1.0, (days_stored / shelf_life) * (temp / 10) * (humidity / 80) * (1 - spoilage_resistance/10))
        has_microbial_spoilage = random.random() < microbial_probability
        
        if has_microbial_spoilage:
            # Select appropriate microorganism for this fruit
            suitable_microbes = []
            for microbe in spoilage_microorganisms:
                # Check if this is a primary target fruit for this microbe
                is_primary_target = fruit_name in microbe[4].get("primary_targets", [])
                # Check if conditions are suitable
                temp_suitable = temp_range[0] <= temp <= temp_range[1]
                humidity_suitable = humidity_range[0] <= humidity <= humidity_range[1]
                pH_suitable = microbe[3][0] <= pH <= microbe[3][1]
                
                # Weight by suitability
                weight = 3.0 if is_primary_target else 1.0
                if temp_suitable: weight *= 1.5
                if humidity_suitable: weight *= 1.5
                if pH_suitable: weight *= 1.5
                
                suitable_microbes.append((microbe, weight))
            
            # Select a microorganism probabilistically
            if suitable_microbes:
                microbes, weights = zip(*suitable_microbes)
                chosen_microbe = random.choices(microbes, weights=weights, k=1)[0]
                
                # Apply microbe-specific effects
                effects = chosen_microbe[4]
                gas_multiplier = effects.get("gas", 1.0)
                voc_multiplier = effects.get("voc", 1.0)
                ph_delta = effects.get("ph", 0)
                
                # Microbial growth increases with favorable conditions
                microbial_growth = calculate_microbial_growth(days_stored, temp, humidity, pH, 
                                                             shelf_life, spoilage_resistance, chosen_microbe)
            else:
                # Generic microbial effects
                gas_multiplier = 1.2
                voc_multiplier = 1.3
                ph_delta = -0.3
                
                # Generic microbial growth
                microbial_growth = calculate_microbial_growth(days_stored, temp, humidity, pH, 
                                                             shelf_life, spoilage_resistance)
            
            # Adjust pH based on microbial activity
            pH = max(3.5, pH + ph_delta * (microbial_growth/100))
        else:
            # No specific microbial spoilage
            gas_multiplier = 1.0
            voc_multiplier = 1.0
            microbial_growth = 0
        
        # Gas emission from respiration and microbial activity
        base_gas = calculate_natural_gas_emission(fruit_name, days_stored, temp, 
                                                  is_climacteric, spoilage_resistance, shelf_life)
        gas = base_gas * gas_multiplier
        
        # VOC production with improved modeling
        voc_base = 75 + random.uniform(-5, 5) + (10 if is_climacteric else 0)
        voc = calculate_voc_production(days_stored, temp, voc_base, is_climacteric, shelf_life) * voc_multiplier
        
        # CO levels rise with increased respiration and microbial activity
        co_base = (respiration_rate / 5) * (1 + 0.1 * days_stored)  # CO production roughly correlated with respiration
        co_spoilage_factor = 1.0 if microbial_growth < 50 else 1.0 + (microbial_growth - 50) / 100
        co = round(co_base * co_spoilage_factor, 1)
        
        # Determine freshness state based on improved criteria
        if forced_state:
            freshness = forced_state
        else:
            # Multiple factors determine freshness
            shelf_life_factor = days_stored / shelf_life
            
            # Temperature accelerates spoilage (higher weight)
            temp_factor = temp / 10.0  # Normalized for weighting
            
            # Humidity effect depends on optimal range
            humidity_optimality = 1.0 - abs((humidity - 85) / 40)  # 85% is typically optimal
            humidity_factor = 1.0 / max(0.5, humidity_optimality)
            
            # Microbial spoilage is critical
            microbial_factor = 1.0 + (microbial_growth / 100)
            
            # Fruit-specific resistance
            resistance_factor = 10.0 / spoilage_resistance
            
            # Combined spoilage score (higher = more spoiled)
            spoilage_score = shelf_life_factor * temp_factor * humidity_factor * microbial_factor * resistance_factor
            
            # Binary classification with improved threshold
            freshness = "spoiled" if spoilage_score > 1.0 else "fresh"
            
            # Force sensor readings to be consistent with freshness label
            if freshness == "spoiled":
                # Ensure spoiled readings look spoiled - maintain distribution shape but ensure values in spoilage range
                gas = max(gas, random.uniform(220, 440)) 
                voc = max(voc, random.uniform(200, 350))
                co = max(co, random.uniform(12, 18))
                pH = min(pH, random.uniform(4.5, 5.8))
                
                # Increase core temperature for spoiled samples
                core_temp = max(core_temp, temp - 0.5)
            else:
                # Ensure fresh readings look fresh
                gas = min(gas, random.uniform(95, 140))
                voc = min(voc, random.uniform(80, 130))
                co = min(co, random.uniform(3.5, 6.0))
                pH = max(pH, random.uniform(6.0, 7.0))
        
        # Additional scientific measurements
        ethylene_level = round(ethylene_production, 2)
        respiration = round(respiration_rate, 2)
        sugar = round(sugar_content, 1)
        moisture = round(moisture_content, 1)
        
        # Return all parameters in the required format
        return [
            sr_no, fruit_name, is_climacteric, temp, humidity, gas, voc, co, core_temp, 
            light, pH, ethylene_level, respiration, sugar, moisture,
            o2_level, co2_level, days_stored, shelf_life, spoilage_resistance, freshness
        ]
    
    # Generate balanced dataset of 2000 samples
    data = []
    sr_no = 1
    
    # Enhanced column names with additional biological parameters
    columns = [
        "Sr_No", "Fruit_Type", "Climacteric", "Ambient_Temp", "Humidity", 
        "Gas", "VOC", "CO", "Core_Temp", "Light", "pH", "Ethylene_Production", 
        "Respiration_Rate", "Sugar_Content", "Moisture_Content", "O2_Level", 
        "CO2_Level", "Days_Stored", "Shelf_Life", "Spoilage_Resistance", "Freshness"
    ]
    
    # Generate 1000 samples of each category with more sophisticated distribution matching
    for category in ["fresh", "spoiled"]:
        samples_needed = 1000
        
        while samples_needed > 0:
            # Select random fruit type - weight climacteric vs non-climacteric based on category
            if category == "fresh":
                # For fresh category, equal chance of any fruit type
                fruit_info = random.choice(fruit_categories)
            else:
                # For spoiled, weight more toward fruits with lower spoilage resistance
                weights = [1/(1 + f[5]/2) for f in fruit_categories]  # Higher weight for less resistant fruits
                fruit_info = random.choices(fruit_categories, weights=weights, k=1)[0]
            
            # Select appropriate storage condition based on category
            if category == "fresh":
                # Fresh fruits more likely to be in proper storage conditions
                storage_weights = [0.7, 0.2, 0.03, 0.05, 0.01, 0.01]
                storage_condition = random.choices(storage_scenarios, weights=storage_weights)[0]
                
                # Days stored is shorter for fresh category
                shelf_life = random.uniform(*fruit_info[6])
                days_stored = random.uniform(1, 0.6 * shelf_life)
            else:
                # Spoiled fruits more likely to be in poor conditions
                storage_weights = [0.05, 0.2, 0.3, 0.25, 0.15, 0.05]
                storage_condition = random.choices(storage_scenarios, weights=storage_weights)[0]
                
                # Days stored is longer for spoiled category
                shelf_life = random.uniform(*fruit_info[6])
                days_stored = random.uniform(0.7 * shelf_life, 1.5 * shelf_life)
            
            # Generate sample
            sample = generate_fruit_sample(sr_no, fruit_info, days_stored, storage_condition, category)
            data.append(sample)
            
            sr_no += 1
            samples_needed -= 1
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Verify distributions to ensure clear separation between fresh and spoiled
    fresh_df = df[df['Freshness'] == 'fresh']
    spoiled_df = df[df['Freshness'] == 'spoiled']
    
    # Check key parameters for good separation
    for param in ['Gas', 'VOC', 'CO', 'pH']:
        fresh_mean = fresh_df[param].mean()
        spoiled_mean = spoiled_df[param].mean()
        fresh_std = fresh_df[param].std()
        spoiled_std = spoiled_df[param].std()
        
        # Calculate separation (distance between means in std dev units)
        separation = abs(fresh_mean - spoiled_mean) / ((fresh_std + spoiled_std) / 2)
        print(f"Parameter {param}: Fresh mean={fresh_mean:.2f}±{fresh_std:.2f}, Spoiled mean={spoiled_mean:.2f}±{spoiled_std:.2f}")
        print(f"Separation: {separation:.2f} std dev units")
        
        # Visualize the separation
        plt.figure(figsize=(10, 6))
        sns.kdeplot(fresh_df[param], label=f'Fresh {param}')
        sns.kdeplot(spoiled_df[param], label=f'Spoiled {param}')
        plt.title(f'Distribution of {param} by Freshness Category')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"param_distribution_{param}.png"))
        plt.close()
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV files
    fresh_df = df[df['Freshness'] == 'fresh']
    spoiled_df = df[df['Freshness'] == 'spoiled']
    
    fresh_path = "enhanced_scientific_fresh_data.csv"
    spoiled_path = "enhanced_scientific_spoiled_data.csv"
    all_path = "enhanced_scientific_spoilage_dataset.csv"
    
    fresh_df.to_csv(os.path.join(output_dir, fresh_path), index=False)
    spoiled_df.to_csv(os.path.join(output_dir, spoiled_path), index=False)
    df.to_csv(os.path.join(output_dir, all_path), index=False)
    
    print(f"✅ Created enhanced scientifically accurate fresh fruit data ({len(fresh_df)} samples)")
    print(f"✅ Created enhanced scientifically accurate spoiled fruit data ({len(spoiled_df)} samples)")
    print(f"✅ Created combined scientific fruit dataset with improved distribution separation ({len(df)} samples)")
    print(f"✅ Dataset includes advanced biological parameters and more realistic spoilage patterns")
    print(f"✅ Files saved to: {output_dir}")
    
    # Create enhanced GAN for better distribution matching
    print("\nTraining specialized GAN for distribution-matched synthetic data generation...")
    
    # Extract core features
    core_features = ['Ambient_Temp', 'Humidity', 'Gas', 'VOC', 'CO', 'Core_Temp', 'Light', 'pH']
    feature_data = df[core_features].values
    
    # Scale the data for GAN
    gan_scaler = MinMaxScaler()
    gan_data_scaled = gan_scaler.fit_transform(feature_data)
    
    # Create labels (fresh=0, spoiled=1)
    gan_labels = np.array([1 if f == 'spoiled' else 0 for f in df['Freshness']])
    gan_onehot = tf.keras.utils.to_categorical(gan_labels, num_classes=2)
    
    # Split into fresh and spoiled for training with equal batches
    fresh_indices = np.where(gan_labels == 0)[0]
    spoiled_indices = np.where(gan_labels == 1)[0]
    
    fresh_features = gan_data_scaled[fresh_indices]
    spoiled_features = gan_data_scaled[spoiled_indices]
    
    # Create specialized GAN model for distribution matching
    latent_dim_bio = 48  # Increased dimension for better modeling
    
    # Distribution-matching GAN
    bio_generator = build_generator(latent_dim_bio, len(core_features), 2)
    bio_discriminator = build_discriminator(len(core_features), 2)
    
    # Mini-training loop for the distribution-matching GAN
    bio_batch_size = 64
    bio_epochs = 2000
    
    # Convert to tensors
    fresh_tensor = tf.convert_to_tensor(fresh_features, dtype=tf.float32)
    spoiled_tensor = tf.convert_to_tensor(spoiled_features, dtype=tf.float32)
    fresh_condition = tf.constant([[1, 0]] * bio_batch_size, dtype=tf.float32)
    spoiled_condition = tf.constant([[0, 1]] * bio_batch_size, dtype=tf.float32)
    
    # Mini optimizers
    bio_g_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    bio_d_opt = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    
    # Train the specialized GAN
    print("Training specialized distribution-matching GAN...")
    for epoch in range(bio_epochs):
        # Get batch of real samples
        fresh_idx = np.random.randint(0, len(fresh_features), bio_batch_size//2)
        spoiled_idx = np.random.randint(0, len(spoiled_features), bio_batch_size//2)
        
        fresh_batch = tf.gather(fresh_tensor, fresh_idx)
        spoiled_batch = tf.gather(spoiled_tensor, spoiled_idx)
        
        # Train on fresh samples
        d_loss_fresh, _, g_loss_fresh, _, _ = train_step(fresh_batch, fresh_condition[:bio_batch_size//2], bio_batch_size//2)
        
        # Train on spoiled samples
        d_loss_spoiled, _, g_loss_spoiled, _, _ = train_step(spoiled_batch, spoiled_condition[:bio_batch_size//2], bio_batch_size//2)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Fresh D loss: {float(d_loss_fresh):.4f}, G loss: {float(g_loss_fresh):.4f}, "
                  f"Spoiled D loss: {float(d_loss_spoiled):.4f}, G loss: {float(g_loss_spoiled):.4f}")
            
    # Generate enhanced GAN samples with better distribution matching
    print("\nGenerating distribution-matched synthetic samples...")
    
    # Generate fresh samples
    fresh_noise = tf.random.normal([1000, latent_dim_bio])
    fresh_condition_vec = tf.constant([[1, 0]] * 1000, dtype=tf.float32)
    fresh_synthetic = bio_generator([fresh_noise, fresh_condition_vec], training=False).numpy()
    fresh_synthetic_rescaled = gan_scaler.inverse_transform(fresh_synthetic)
    
    # Generate spoiled samples
    spoiled_noise = tf.random.normal([1000, latent_dim_bio])
    spoiled_condition_vec = tf.constant([[0, 1]] * 1000, dtype=tf.float32)
    spoiled_synthetic = bio_generator([spoiled_noise, spoiled_condition_vec], training=False).numpy()
    spoiled_synthetic_rescaled = gan_scaler.inverse_transform(spoiled_synthetic)
    
    # Create dataframes
    gan_fresh_df = pd.DataFrame(fresh_synthetic_rescaled, columns=core_features)
    gan_fresh_df['Freshness'] = 'fresh'
    gan_fresh_df['Source'] = 'GAN-Enhanced'
    
    gan_spoiled_df = pd.DataFrame(spoiled_synthetic_rescaled, columns=core_features)
    gan_spoiled_df['Freshness'] = 'spoiled'
    gan_spoiled_df['Source'] = 'GAN-Enhanced'
    
    # Verify improved distribution matching
    print("\nVerifying distribution matching quality...")
    
    # Compare real vs GAN fresh distributions
    for feature in core_features:
        # Calculate KS statistic
        ks_stat_fresh, p_val_fresh = ks_2samp(fresh_df[feature], gan_fresh_df[feature])
        ks_stat_spoiled, p_val_spoiled = ks_2samp(spoiled_df[feature], gan_spoiled_df[feature])
        
        print(f"Feature {feature}:")
        print(f"  Fresh KS test: stat={ks_stat_fresh:.4f}, p-value={p_val_fresh:.4f}")
        print(f"  Spoiled KS test: stat={ks_stat_spoiled:.4f}, p-value={p_val_spoiled:.4f}")
        
        # Visualize distributions
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.kdeplot(fresh_df[feature], label=f'Real Fresh')
        sns.kdeplot(gan_fresh_df[feature], label=f'GAN Fresh')
        plt.title(f'Fresh {feature} Distribution Matching')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        sns.kdeplot(spoiled_df[feature], label=f'Real Spoiled')
        sns.kdeplot(gan_spoiled_df[feature], label=f'GAN Spoiled')
        plt.title(f'Spoiled {feature} Distribution Matching')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_match_{feature}.png"))
        plt.close()
    
    # Save GAN-generated data
    gan_combined_df = pd.concat([gan_fresh_df, gan_spoiled_df])
    gan_combined_df.to_csv(os.path.join(output_dir, "distribution_matched_gan_data.csv"), index=False)
    
    print(f"✅ Generated {len(gan_combined_df)} distribution-matched GAN samples")
    print(f"✅ Distribution comparison plots saved to {output_dir}/")
    
    return df, gan_combined_df

# Run analysis and generate data
try:
    # Plot enhanced training history
    plot_training_history(history)
    
    # Evaluate overall synthetic data quality
    real_df_for_eval = real_data.drop('freshness', axis=1)
    _, _, similarity_score = evaluate_synthetic_data(real_df_for_eval, validated_data.sample(min(1000, len(validated_data))))
    
    # Add freshness labels to each category's data
    fresh_data['freshness'] = 'fresh'
    spoiled_data['freshness'] = 'spoiled'
    
    # Save all datasets
    fresh_data.to_csv(f"{output_dir}/fresh_food_data.csv", index=False)
    spoiled_data.to_csv(f"{output_dir}/spoiled_food_data.csv", index=False)
    validated_data.sample(1000).to_csv(f"{output_dir}/general_synthetic_food_data.csv", index=False)
    
    # Create combined dataset with samples from each category
    final_synthetic_data = pd.concat([
        validated_data.sample(min(1000, len(validated_data))),
        fresh_data,
        spoiled_data
    ])
    
    # Save the final combined dataset
    final_synthetic_data.to_csv(f"{output_dir}/synthetic_food_freshness_data.csv", index=False)
    
    # Create and save the scientifically accurate fruit spoilage dataset (2000 samples)
    scientific_data, gan_enhanced_data = create_fruit_spoilage_dataset()
    
    print(f"\n✅ All synthetic food sensor data saved to {output_dir}/ directory")
    print(f"✅ Generated {len(final_synthetic_data)} total synthetic samples")
    print(f"✅ Generated {len(fresh_data)} fresh food samples")
    print(f"✅ Generated {len(spoiled_data)} spoiled food samples")
    print(f"✅ Created comprehensive 2000-entry scientific dataset")
    print(f"✅ Overall statistical similarity: {100-similarity_score:.2f}% (higher is better)")
    
except Exception as e:
    print(f"Error in final processing: {str(e)}")
