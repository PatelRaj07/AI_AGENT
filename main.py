import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import os
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class AdvancedRPSModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedRPSModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        lstm_out = lstm_out[:, -1, :]  # Get the last time step output
        x = self.dropout(lstm_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.hidden_size),
                torch.zeros(2, batch_size, self.hidden_size))

class AdvancedRockPaperScissorsAI:
    def __init__(self, model_path=None):
        self.choices = ['rock', 'paper', 'scissors']
        self.move_to_index = {'rock': 0, 'paper': 1, 'scissors': 2}
        self.index_to_move = {0: 'rock', 1: 'paper', 2: 'scissors'}
        self.winning_move = {'rock': 'paper', 'paper': 'scissors', 'scissors': 'rock'}
        
        # Game statistics
        self.ai_score = 0
        self.user_score = 0
        self.tie_score = 0
        self.games_played = 0
        
        # Sequence history
        self.sequence_length = 10  # Length of sequence to consider for predictions
        self.user_history = deque(maxlen=1000)  # Store longer history for analysis
        self.ai_history = deque(maxlen=1000)
        self.results_history = deque(maxlen=1000)
        
        # Neural network parameters
        self.input_size = 9  # 3 one-hot encoded moves for user, 3 for AI, 3 for result
        self.hidden_size = 128
        self.output_size = 3
        self.batch_size = 1
        
        # Create or load the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AdvancedRPSModel(self.input_size, self.hidden_size, self.output_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Reinforcement learning parameters
        self.memory = deque(maxlen=2000)  # Experience replay memory
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Multiple strategies
        self.strategies = {
            'neural_net': self._predict_neural_net,
            'markov': self._predict_markov,
            'frequency': self._predict_frequency,
            'psychology': self._predict_psychology,
            'ensemble': self._predict_ensemble
        }
        
        # Markov models (1st, 2nd, and 3rd order)
        self.markov_first = np.ones((3, 3))
        self.markov_second = np.ones((9, 3))
        self.markov_third = np.ones((27, 3))
        
        # Pattern counters
        self.pattern_lengths = [3, 4, 5, 6]  # Look for patterns of these lengths
        self.pattern_database = {}
        
        # Meta-learning - strategy effectiveness tracking
        self.strategy_performance = {strategy: {'wins': 0, 'total': 0} for strategy in self.strategies}
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Visualization data
        self.win_rates_over_time = []
        self.strategy_usage = {strategy: 0 for strategy in self.strategies}
        
        # Auto save settings
        self.auto_save_interval = 50  # Save every 50 games
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    def get_ai_move(self):
        if len(self.user_history) < self.sequence_length:
            # Not enough history yet, choose randomly but with some psychology
            return self._predict_psychology()
        
        # Dynamic strategy selection based on performance
        strategy = self._select_best_strategy()
        predicted_move = self.strategies[strategy]()
        
        # Record which strategy was used
        self.strategy_usage[strategy] += 1
        
        return predicted_move
    
    def _select_best_strategy(self):
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(list(self.strategies.keys()))
        
        # Calculate win rates for each strategy
        strategy_scores = {}
        for strategy, stats in self.strategy_performance.items():
            if stats['total'] == 0:
                strategy_scores[strategy] = 0
            else:
                strategy_scores[strategy] = stats['wins'] / stats['total']
        
        # Weighted random selection based on performance
        total_score = sum(strategy_scores.values())
        if total_score == 0:
            return 'ensemble'  # Default to ensemble if no data
        
        # Add small probability to each strategy to ensure exploration
        probabilities = {s: (score + 0.1) / (total_score + 0.1 * len(strategy_scores)) 
                         for s, score in strategy_scores.items()}
        
        strategies = list(probabilities.keys())
        probs = [probabilities[s] for s in strategies]
        return random.choices(strategies, weights=probs, k=1)[0]

    def _predict_neural_net(self):
        # Prepare input sequence
        sequence = self._prepare_sequence()
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get prediction from neural network
        with torch.no_grad():
            self.model.eval()
            output, _ = self.model(sequence_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            
        # Apply strategic thinking to counter the predicted move
        predicted_user_move = self.index_to_move[np.argmax(probs)]
        return self.winning_move[predicted_user_move]

    def _predict_markov(self):
        # Combine predictions from 1st, 2nd, and 3rd order Markov models
        if len(self.user_history) < 3:
            idx = random.randint(0, 2)
            return self.index_to_move[idx]
        
        # 1st order prediction
        last_move = self.user_history[-1]
        last_idx = self.move_to_index[last_move]
        first_order_probs = self.markov_first[last_idx] / np.sum(self.markov_first[last_idx])
        
        # 2nd order prediction
        second_last_move = self.user_history[-2]
        second_last_idx = self.move_to_index[second_last_move]
        combined_idx_2 = second_last_idx * 3 + last_idx
        second_order_probs = self.markov_second[combined_idx_2] / np.sum(self.markov_second[combined_idx_2])
        
        # 3rd order prediction
        third_last_move = self.user_history[-3]
        third_last_idx = self.move_to_index[third_last_move]
        combined_idx_3 = third_last_idx * 9 + combined_idx_2
        third_order_probs = self.markov_third[combined_idx_3] / np.sum(self.markov_third[combined_idx_3])
        
        # Weight the predictions (give more weight to higher order models)
        combined_probs = 0.2 * first_order_probs + 0.3 * second_order_probs + 0.5 * third_order_probs
        predicted_idx = np.argmax(combined_probs)
        predicted_move = self.index_to_move[predicted_idx]
        
        return self.winning_move[predicted_move]

    def _predict_frequency(self):
        # Analyze frequency of moves in different contexts
        if len(self.user_history) < 5:
            return random.choice(self.choices)
        
        # Overall frequency
        move_counts = {move: 0 for move in self.choices}
        for move in self.user_history:
            move_counts[move] += 1
        
        # Recent frequency (last 10 moves)
        recent_moves = list(self.user_history)[-10:]
        recent_counts = {move: 0 for move in self.choices}
        for move in recent_moves:
            recent_counts[move] += 1
        
        # After-win frequency
        after_win_counts = {move: 0.1 for move in self.choices}  # Laplace smoothing
        for i in range(1, len(self.results_history)):
            if self.results_history[i-1] == "User wins!":
                after_win_counts[self.user_history[i]] += 1
        
        # After-loss frequency
        after_loss_counts = {move: 0.1 for move in self.choices}  # Laplace smoothing
        for i in range(1, len(self.results_history)):
            if self.results_history[i-1] == "AI wins!":
                after_loss_counts[self.user_history[i]] += 1
        
        # Combine the frequencies with weighting
        combined_frequencies = {}
        for move in self.choices:
            combined_frequencies[move] = (
                0.1 * move_counts[move] / max(sum(move_counts.values()), 1) +
                0.3 * recent_counts[move] / max(len(recent_moves), 1)
            )
            
            # Add context-specific frequencies
            if self.results_history and self.results_history[-1] == "User wins!":
                combined_frequencies[move] += 0.3 * (after_win_counts[move] / sum(after_win_counts.values()))
            elif self.results_history and self.results_history[-1] == "AI wins!":
                combined_frequencies[move] += 0.3 * (after_loss_counts[move] / sum(after_loss_counts.values()))
        
        predicted_move = max(combined_frequencies, key=combined_frequencies.get)
        return self.winning_move[predicted_move]

    def _predict_psychology(self):
        """Psychological strategy based on game theory and human behavior patterns"""
        if not self.user_history:
            # First move - people often start with rock
            return "paper"
        
        # Check for repeated moves
        if len(self.user_history) >= 3:
            last_three = list(self.user_history)[-3:]
            if last_three[0] == last_three[1] == last_three[2]:
                # User repeating the same move - they might change
                repeated_move = last_three[0]
                # Predict they'll switch to move that would beat their repeated move
                likely_switch = self.winning_move[repeated_move]
                # Counter the switch
                return self.winning_move[likely_switch]
        
        # After losing twice, users often switch to what would have beaten the AI's last move
        if len(self.results_history) >= 2:
            last_two_results = list(self.results_history)[-2:]
            if last_two_results[0] == last_two_results[1] == "AI wins!":
                ai_last_move = list(self.ai_history)[-1]
                predicted_user_move = self.winning_move[ai_last_move]
                return self.winning_move[predicted_user_move]
        
        # After winning, users often stick with their winning move
        if self.results_history and self.results_history[-1] == "User wins!":
            winning_move = list(self.user_history)[-1]
            return self.winning_move[winning_move]  # Counter their likely repeat
        
        # Pattern "rock, paper" often leads to "scissors" as humans try to complete the sequence
        if len(self.user_history) >= 2:
            last_two = list(self.user_history)[-2:]
            if last_two == ["rock", "paper"]:
                return "rock"  # Counter the likely "scissors"
            elif last_two == ["paper", "scissors"]:
                return "paper"  # Counter the likely "rock"
            elif last_two == ["scissors", "rock"]:
                return "scissors"  # Counter the likely "paper"
        
        # Fallback - slightly weighted towards paper (as rock is most common first move)
        moves = ["rock", "paper", "scissors", "paper"]
        return random.choice(moves)

    def _predict_ensemble(self):
        """Combine predictions from all strategies"""
        predictions = {
            'neural_net': self._predict_neural_net(),
            'markov': self._predict_markov(),
            'frequency': self._predict_frequency(),
            'psychology': self._predict_psychology()
        }
        
        # Count votes for each move
        votes = {move: 0 for move in self.choices}
        for strategy, move in predictions.items():
            votes[move] += self.strategy_performance[strategy]['wins'] / max(self.strategy_performance[strategy]['total'], 1)
        
        # If all strategies have same weight, add a small random factor
        if len(set(votes.values())) == 1:
            for move in votes:
                votes[move] += random.random() * 0.1
        
        # Return move with most votes
        return max(votes, key=votes.get)

    def _prepare_sequence(self):
        """Create input sequence for neural network"""
        # Take the last 'sequence_length' interactions
        sequence = []
        
        # If we don't have enough history, pad with zeros
        padding_needed = max(0, self.sequence_length - len(self.user_history))
        
        # Add padding
        for _ in range(padding_needed):
            # [user_move, ai_move, result] all zeros (one-hot encoded)
            sequence.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Add actual history
        for i in range(max(0, len(self.user_history) - self.sequence_length), len(self.user_history)):
            # One-hot encode user move
            user_move = [0, 0, 0]
            user_move[self.move_to_index[self.user_history[i]]] = 1
            
            # One-hot encode AI move
            ai_move = [0, 0, 0]
            if i < len(self.ai_history):
                ai_move[self.move_to_index[self.ai_history[i]]] = 1
            
            # One-hot encode result
            result = [0, 0, 0]  # [tie, ai win, user win]
            if i < len(self.results_history):
                if self.results_history[i] == "It's a tie!":
                    result[0] = 1
                elif self.results_history[i] == "AI wins!":
                    result[1] = 1
                else:  # User wins
                    result[2] = 1
            
            # Combine all features
            sequence.append(user_move + ai_move + result)
        
        return sequence

    def update_model(self, user_move, ai_move, result):
        """Update all models with the new information"""
        # Update history
        self.user_history.append(user_move)
        self.ai_history.append(ai_move)
        self.results_history.append(result)
        
        # Update Markov models
        if len(self.user_history) >= 2:
            # Update 1st order
            prev_move = list(self.user_history)[-2]
            prev_idx = self.move_to_index[prev_move]
            curr_idx = self.move_to_index[user_move]
            self.markov_first[prev_idx][curr_idx] += 1
        
        if len(self.user_history) >= 3:
            # Update 2nd order
            second_prev_move = list(self.user_history)[-3]
            prev_move = list(self.user_history)[-2]
            second_prev_idx = self.move_to_index[second_prev_move]
            prev_idx = self.move_to_index[prev_move]
            curr_idx = self.move_to_index[user_move]
            
            combined_idx_2 = second_prev_idx * 3 + prev_idx
            self.markov_second[combined_idx_2][curr_idx] += 1
        
        if len(self.user_history) >= 4:
            # Update 3rd order
            third_prev_move = list(self.user_history)[-4]
            second_prev_move = list(self.user_history)[-3]
            prev_move = list(self.user_history)[-2]
            
            third_prev_idx = self.move_to_index[third_prev_move]
            second_prev_idx = self.move_to_index[second_prev_move]
            prev_idx = self.move_to_index[prev_move]
            curr_idx = self.move_to_index[user_move]
            
            combined_idx_2 = second_prev_idx * 3 + prev_idx
            combined_idx_3 = third_prev_idx * 9 + combined_idx_2
            self.markov_third[combined_idx_3][curr_idx] += 1
        
        # Update pattern database
        for length in self.pattern_lengths:
            if len(self.user_history) >= length + 1:
                pattern = tuple(list(self.user_history)[-(length+1):-1])
                if pattern not in self.pattern_database:
                    self.pattern_database[pattern] = {move: 0 for move in self.choices}
                self.pattern_database[pattern][user_move] += 1
        
        # Train neural network
        if len(self.user_history) >= self.sequence_length + 1:
            self._train_neural_network()
        
        # Update reinforcement learning
        if len(self.user_history) >= 2:
            self._update_reinforcement_learning(result)
        
        # Update strategy performance
        if hasattr(self, '_last_strategy'):
            win = 1 if result == "AI wins!" else 0
            self.strategy_performance[self._last_strategy]['wins'] += win
            self.strategy_performance[self._last_strategy]['total'] += 1
        
        # Update win rate tracking
        if self.games_played % 10 == 0:  # Record every 10 games
            if self.games_played > 0:
                win_rate = self.ai_score / self.games_played
                self.win_rates_over_time.append(win_rate)
        
        # Auto-save
        if self.games_played % self.auto_save_interval == 0 and self.games_played > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_model(f"{self.model_dir}/rps_model_{timestamp}.pkl")

    def _train_neural_network(self):
        """Train the neural network on the latest data"""
        if len(self.user_history) <= self.sequence_length:
            return
        
        self.model.train()
        sequence = self._prepare_sequence()
        
        # The input is the sequence up to the last move
        x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
        
        # The target is the user's next move (the latest one)
        y = torch.LongTensor([self.move_to_index[self.user_history[-1]]]).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output, _ = self.model(x)
        loss = self.criterion(output, y)
        
        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

    def _update_reinforcement_learning(self, result):
        """Update reinforcement learning parameters"""
        # Convert result to reward
        if result == "AI wins!":
            reward = 1.0
        elif result == "It's a tie!":
            reward = 0.3
        else:
            reward = -1.0
        
        # Store experience in memory
        if len(self.user_history) >= self.sequence_length + 1:
            current_state = self._prepare_sequence()[:-1]  # State before latest move
            next_state = self._prepare_sequence()  # State after latest move
            
            # Convert AI move to action (index)
            action = self.move_to_index[self.ai_history[-1]]
            
            # Store the experience
            self.memory.append((current_state, action, reward, next_state))
        
        # Decrease epsilon (exploration rate)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Train on a mini-batch from memory
        self._replay(32)

    def _replay(self, batch_size):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        # Sample random experiences
        minibatch = random.sample(self.memory, batch_size)
        
        self.model.train()
        for state, action, reward, next_state in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            
            # Q-learning update
            with torch.no_grad():
                next_q_values, _ = self.model(next_state)
                max_next_q = torch.max(next_q_values).item()
                target = reward + self.gamma * max_next_q
            
            # Get current Q values
            current_q_values, _ = self.model(state)
            current_q = current_q_values[0][action].item()
            
            # Compute loss (simplified)
            target_f = current_q_values.clone()
            target_f[0][action] = target
            
            # Train
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, torch.argmax(target_f, dim=1))
            loss.backward()
            self.optimizer.step()

    def play_round(self, user_move):
        if user_move not in self.choices:
            return "Invalid move. Choose rock, paper, or scissors."
        
        ai_move = self.get_ai_move()
        self.games_played += 1
        
        # Determine winner
        result = self._determine_winner(user_move, ai_move)
        
        # Update models
        self.update_model(user_move, ai_move, result)
        
        # Return results
        return {
            'ai_move': ai_move,
            'result': result,
            'ai_score': self.ai_score,
            'user_score': self.user_score,
            'tie_score': self.tie_score,
            'games_played': self.games_played,
            'current_pattern': self._get_pattern_analysis()
        }
    
    def _determine_winner(self, user_move, ai_move):
        if user_move == ai_move:
            self.tie_score += 1
            return "It's a tie!"
        elif self.winning_move[user_move] == ai_move:
            self.ai_score += 1
            return "AI wins!"
        else:
            self.user_score += 1
            return "You win!"
    
    def _get_pattern_analysis(self):
        """Return a brief analysis of the user's patterns"""
        if len(self.user_history) < 10:
            return "Still analyzing your play style..."
        
        # Check for move preference
        move_counts = {move: 0 for move in self.choices}
        for move in self.user_history:
            move_counts[move] += 1
        total = sum(move_counts.values())
        
        favorite_move = max(move_counts, key=move_counts.get)
        favorite_pct = move_counts[favorite_move] / total * 100
        
        if favorite_pct > 40:
            return f"You seem to favor {favorite_move} ({favorite_pct:.1f}% of the time)"
        
        # Check for patterns after wins/losses
        win_followed_by = {move: 0 for move in self.choices}
        loss_followed_by = {move: 0 for move in self.choices}
        
        for i in range(1, len(self.results_history)):
            if self.results_history[i-1] == "User wins!":
                win_followed_by[self.user_history[i]] += 1
            elif self.results_history[i-1] == "AI wins!":
                loss_followed_by[self.user_history[i]] += 1
        
        win_total = sum(win_followed_by.values())
        loss_total = sum(loss_followed_by.values())
        
        if win_total > 0:
            win_favorite = max(win_followed_by, key=win_followed_by.get)
            win_pct = win_followed_by[win_favorite] / win_total * 100
            if win_pct > 50:
                return f"After winning, you play {win_favorite} {win_pct:.1f}% of the time"
        
        if loss_total > 0:
            loss_favorite = max(loss_followed_by, key=loss_followed_by.get)
            loss_pct = loss_followed_by[loss_favorite] / loss_total * 100
            if loss_pct > 50:
                return f"After losing, you play {loss_favorite} {loss_pct:.1f}% of the time"
        
        return "Your play style is becoming more unpredictable..."

    def get_stats(self):
        """Return detailed stats about the game"""
        if self.games_played == 0:
            return "No games played yet."
        
        ai_win_rate = (self.ai_score / self.games_played) * 100
        user_win_rate = (self.user_score / self.games_played) * 100
        tie_rate = (self.tie_score / self.games_played) * 100
        
        stats = {
            'games_played': self.games_played,
            'ai_score': self.ai_score,
            'user_score': self.user_score,
            'tie_score': self.tie_score,
            'ai_win_rate': ai_win_rate,
            'user_win_rate': user_win_rate,
            'tie_rate': tie_rate,
            'strategy_usage': self.strategy_usage,
            'exploration_rate': self.epsilon,
            'pattern_analysis': self._get_pattern_analysis()
        }
        
        return stats

    def save_model(self, filepath):
        """Save the AI model and all its state"""
        # Create dictionary with all important state
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'user_history': list(self.user_history),
            'ai_history': list(self.ai_history),
            'results_history': list(self.results_history),
            'markov_first': self.markov_first,
            'markov_second': self.markov_second,
            'markov_third': self.markov_third,
            'pattern_database': self.pattern_database,
            'strategy_performance': self.strategy_performance,
            'ai_score': self.ai_score,
            'user_score': self.user_score,
            'tie_score': self.tie_score,
            'games_played': self.games_played,
            'epsilon': self.epsilon,
            'win_rates_over_time': self.win_rates_over_time,
            'strategy_usage': self.strategy_usage
        }
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load the AI model and all its state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Load model parameters
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        # Load history
        self.user_history = deque(state['user_history'], maxlen=1000)
        self.ai_history = deque(state['ai_history'], maxlen=1000)
        self.results_history = deque(state['results_history'], maxlen=1000)
        
        # Load Markov models
        self.markov_first = state['markov_first']
        self.markov_second = state['markov_second']
        self.markov_third = state['markov_third']
        
        # Load pattern database
        self.pattern_database = state['pattern_database']
        
        # Load strategy performance
        self.strategy_performance = state['strategy_performance']
        
        # Load game stats
        self.ai_score = state['ai_score']
        self.user_score = state['user_score']
        self.tie_score = state['tie_score']
        self.games_played = state['games_played']
        self.epsilon = state['epsilon']
        
        # Load visualization data
        self.win_rates_over_time = state['win_rates_over_time']
        self.strategy_usage = state['strategy_usage']
        
        print(f"Model loaded from {filepath}")

    def visualize_performance(self):
        """Generate visualizations of AI performance"""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Win rate over time
        plt.subplot(2, 2, 1)
        plt.plot(self.win_rates_over_time)
        plt.title('AI Win Rate Over Time')
        plt.xlabel('Batches of 10 Games')
        plt.ylabel('Win Rate')
        plt.grid(True)
        
        # Plot 2: Overall game outcomes
        plt.subplot(2, 2, 2)
        labels = ['AI Wins', 'User Wins', 'Ties']
        sizes = [self.ai_score, self.user_score, self.tie_score]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Game Outcomes')
        
        # Plot 3: Strategy usage
        plt.subplot(2, 2, 3)
        strategies = list(self.strategy_usage.keys())
        usage = list(self.strategy_usage.values())
        plt.bar(strategies, usage)
        plt.title('Strategy Usage')
        plt.xticks(rotation=45)
        
        # Plot 4: User move distribution
        plt.subplot(2, 2, 4)
        move_counts = {move: 0 for move in self.choices}
        for move in self.user_history:
            move_counts[move] += 1
        moves = list(move_counts.keys())
        counts = list(move_counts.values())
        plt.bar(moves, counts)
        plt.title('User Move Distribution')
        
        plt.tight_layout()
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"rps_stats_{timestamp}.png")
        plt.close()
        
        return f"rps_stats_{timestamp}.png"

    def get_prediction_probabilities(self):
        """Return the probability distribution for the user's next move"""
        if len(self.user_history) < self.sequence_length:
            # Not enough history, use simple psychology-based probabilities
            if not self.user_history:
                # First move - people often start with rock
                return {'rock': 0.5, 'paper': 0.3, 'scissors': 0.2}
            else:
                # Simple probabilities based on psychology
                last_move = self.user_history[-1]
                if last_move == 'rock':
                    return {'rock': 0.3, 'paper': 0.3, 'scissors': 0.4}
                elif last_move == 'paper':
                    return {'rock': 0.2, 'paper': 0.3, 'scissors': 0.5}
                else:  # scissors
                    return {'rock': 0.5, 'paper': 0.2, 'scissors': 0.3}
        
        # Get probabilities from neural network
        sequence = self._prepare_sequence()
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            output, _ = self.model(sequence_tensor)
            nn_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # Get probabilities from Markov model
        markov_probs = self._get_markov_probabilities()
        
        # Combine probabilities (weighted average)
        combined_probs = {}
        for i, move in enumerate(self.choices):
            combined_probs[move] = 0.6 * nn_probs[i] + 0.4 * markov_probs[move]
        
        return combined_probs
    
    def _get_markov_probabilities(self):
        """Get probabilities from Markov models"""
        if len(self.user_history) < 3:
            return {move: 1/3 for move in self.choices}
        
        # 1st order prediction
        last_move = self.user_history[-1]
        last_idx = self.move_to_index[last_move]
        first_order_probs = self.markov_first[last_idx] / np.sum(self.markov_first[last_idx])
        
        # 2nd order prediction
        second_last_move = self.user_history[-2]
        second_last_idx = self.move_to_index[second_last_move]
        combined_idx_2 = second_last_idx * 3 + last_idx
        second_order_probs = self.markov_second[combined_idx_2] / np.sum(self.markov_second[combined_idx_2])
        
        # 3rd order prediction
        third_last_move = self.user_history[-3]
        third_last_idx = self.move_to_index[third_last_move]
        combined_idx_3 = third_last_idx * 9 + combined_idx_2
        third_order_probs = self.markov_third[combined_idx_3] / np.sum(self.markov_third[combined_idx_3])
        
        # Weight the predictions
        combined_probs = {}
        for i, move in enumerate(self.choices):
            combined_probs[move] = (
                0.2 * first_order_probs[i] + 
                0.3 * second_order_probs[i] + 
                0.5 * third_order_probs[i]
            )
        
        return combined_probs


class RockPaperScissorsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Rock Paper Scissors AI")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Create the AI agent
        self.ai = AdvancedRockPaperScissorsAI()
        
        # Set up the UI
        self._setup_ui()
        
        # For tracking last prediction
        self.last_prediction = None
        
        # Update prediction bars initially
        self.update_prediction_bars()
        
    def _setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced Rock Paper Scissors AI", 
                                font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Game controls frame
        game_frame = ttk.Frame(main_frame)
        game_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left side - Game controls
        controls_frame = ttk.LabelFrame(game_frame, text="Game Controls", padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Move buttons frame
        move_frame = ttk.Frame(controls_frame)
        move_frame.pack(fill=tk.X, pady=10)
        
        # Button style
        style = ttk.Style()
        style.configure("Move.TButton", font=("Arial", 12, "bold"), padding=10)
        
        # Rock button with image
        self.rock_button = ttk.Button(move_frame, text="Rock ✊", style="Move.TButton",
                                     command=lambda: self.play_move("rock"))
        self.rock_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Paper button with image
        self.paper_button = ttk.Button(move_frame, text="Paper ✋", style="Move.TButton",
                                      command=lambda: self.play_move("paper"))
        self.paper_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Scissors button with image
        self.scissors_button = ttk.Button(move_frame, text="Scissors ✌️", style="Move.TButton",
                                        command=lambda: self.play_move("scissors"))
        self.scissors_button.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Score display
        score_frame = ttk.Frame(controls_frame)
        score_frame.pack(fill=tk.X, pady=10)
        
        # Score labels
        self.ai_score_label = ttk.Label(score_frame, text="AI: 0", font=("Arial", 14))
        self.ai_score_label.pack(side=tk.LEFT, expand=True)
        
        self.user_score_label = ttk.Label(score_frame, text="You: 0", font=("Arial", 14))
        self.user_score_label.pack(side=tk.LEFT, expand=True)
        
        self.tie_score_label = ttk.Label(score_frame, text="Ties: 0", font=("Arial", 14))
        self.tie_score_label.pack(side=tk.LEFT, expand=True)
        
        # Game result display
        result_frame = ttk.Frame(controls_frame)
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = ttk.Label(result_frame, text="Make your move!", 
                                     font=("Arial", 14, "bold"))
        self.result_label.pack(fill=tk.X)
        
        # AI observation display
        observation_frame = ttk.Frame(controls_frame)
        observation_frame.pack(fill=tk.X, pady=10)
        
        self.observation_label = ttk.Label(observation_frame, 
                                          text="AI is analyzing your play style...",
                                          font=("Arial", 12, "italic"), wraplength=400)
        self.observation_label.pack(fill=tk.X)
        
        # Command buttons
        cmd_frame = ttk.Frame(controls_frame)
        cmd_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(cmd_frame, text="Save AI", command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(cmd_frame, text="Load AI", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(cmd_frame, text="Statistics", command=self.show_stats).pack(side=tk.LEFT, padx=5)
        ttk.Button(cmd_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        
        # Right side - Prediction display
        prediction_frame = ttk.LabelFrame(game_frame, text="AI's Prediction of Your Next Move", padding="10")
        prediction_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Prediction probability bars
        self.probability_frame = ttk.Frame(prediction_frame)
        self.probability_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create Matplotlib figure for probability bars
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.probability_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Game history
        history_frame = ttk.LabelFrame(main_frame, text="Game History", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for game history
        columns = ("round", "your_move", "ai_move", "result")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        
        # Define headings
        self.history_tree.heading("round", text="Round")
        self.history_tree.heading("your_move", text="Your Move")
        self.history_tree.heading("ai_move", text="AI Move")
        self.history_tree.heading("result", text="Result")
        
        # Define columns width
        self.history_tree.column("round", width=50)
        self.history_tree.column("your_move", width=150)
        self.history_tree.column("ai_move", width=150)
        self.history_tree.column("result", width=150)
        
        # Add a scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        # Pack the treeview and scrollbar
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to play!")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def update_prediction_bars(self):
        """Update the prediction probability bars"""
        # Get prediction probabilities from AI
        probabilities = self.ai.get_prediction_probabilities()
        self.last_prediction = probabilities
        
        # Clear the plot
        self.ax.clear()
        
        # Set up the bar chart
        moves = list(probabilities.keys())
        probs = [probabilities[move] for move in moves]
        
        # Create horizontal bar chart
        bars = self.ax.barh(moves, probs, color=['#3498db', '#2ecc71', '#e74c3c'])
        
        # Add percentage labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            self.ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{width:.1%}', va='center')
        
        # Set title and limits
        self.ax.set_title("Probability of Your Next Move")
        self.ax.set_xlim(0, 1)
        
        # Update the canvas
        self.canvas.draw()
    
    def play_move(self, user_move):
        """Handle the user's move"""
        # Disable buttons briefly to prevent spam clicking
        self.rock_button.state(['disabled'])
        self.paper_button.state(['disabled'])
        self.scissors_button.state(['disabled'])
        
        # Update status
        self.status_var.set(f"Processing your move: {user_move}...")
        self.root.update()
        
        # Get AI's move and result
        result = self.ai.play_round(user_move)
        
        # Update score labels
        self.ai_score_label.config(text=f"AI: {result['ai_score']}")
        self.user_score_label.config(text=f"You: {result['user_score']}")
        self.tie_score_label.config(text=f"Ties: {result['tie_score']}")
        
        # Update result label
        self.result_label.config(text=f"You chose {user_move}, AI chose {result['ai_move']}. {result['result']}")
        
        # Update observation
        if result['current_pattern']:
            self.observation_label.config(text=result['current_pattern'])
        
        # Add to history
        self.history_tree.insert("", 0, values=(
            result['games_played'],
            user_move,
            result['ai_move'],
            result['result']
        ))
        
        # Update prediction bars in a separate thread to avoid UI freezing
        threading.Thread(target=self.update_prediction_bars).start()
        
        # Re-enable buttons
        self.rock_button.state(['!disabled'])
        self.paper_button.state(['!disabled'])
        self.scissors_button.state(['!disabled'])
        
        # Update status
        self.status_var.set("Ready for your next move!")
    
    def save_model(self):
        """Save the AI model"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=self.ai.model_dir
        )
        if filename:
            try:
                self.ai.save_model(filename)
                self.status_var.set(f"Model saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {e}")
    
    def load_model(self):
        """Load an AI model"""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=self.ai.model_dir
        )
        if filename:
            try:
                self.ai.load_model(filename)
                
                # Update UI to reflect loaded state
                self.ai_score_label.config(text=f"AI: {self.ai.ai_score}")
                self.user_score_label.config(text=f"You: {self.ai.user_score}")
                self.tie_score_label.config(text=f"Ties: {self.ai.tie_score}")
                
                # Clear history and re-populate
                self.history_tree.delete(*self.history_tree.get_children())
                for i in range(min(len(self.ai.user_history), len(self.ai.ai_history), len(self.ai.results_history))):
                    self.history_tree.insert("", 0, values=(
                        len(self.ai.user_history) - i,
                        self.ai.user_history[-(i+1)],
                        self.ai.ai_history[-(i+1)],
                        self.ai.results_history[-(i+1)]
                    ))
                
                # Update observation
                pattern_analysis = self.ai._get_pattern_analysis()
                self.observation_label.config(text=pattern_analysis)
                
                # Update prediction bars
                self.update_prediction_bars()
                
                self.status_var.set(f"Model loaded from {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def show_stats(self):
        """Show detailed statistics"""
        stats = self.ai.get_stats()
        
        # Create a new window for stats
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Game Statistics")
        stats_window.geometry("600x500")
        
        # Create a frame for the stats
        stats_frame = ttk.Frame(stats_window, padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add statistics
        row = 0
        ttk.Label(stats_frame, text="Game Statistics", font=("Arial", 16, "bold")).grid(
            row=row, column=0, columnspan=2, pady=10, sticky=tk.W)
        
        row += 1
        ttk.Label(stats_frame, text="Games Played:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=str(stats['games_played'])).grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="AI Wins:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{stats['ai_score']} ({stats['ai_win_rate']:.1f}%)").grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="User Wins:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{stats['user_score']} ({stats['user_win_rate']:.1f}%)").grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="Ties:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{stats['tie_score']} ({stats['tie_rate']:.1f}%)").grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="Exploration Rate:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=f"{stats['exploration_rate']:.3f}").grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="Pattern Analysis:").grid(
            row=row, column=0, sticky=tk.W, pady=2)
        ttk.Label(stats_frame, text=stats['pattern_analysis'], wraplength=300).grid(
            row=row, column=1, sticky=tk.W, pady=2)
        
        row += 1
        ttk.Label(stats_frame, text="Strategy Usage:", font=("Arial", 12, "bold")).grid(
            row=row, column=0, columnspan=2, pady=(10, 5), sticky=tk.W)
        
        # Add strategy usage
        row += 1
        for strategy, count in stats['strategy_usage'].items():
            ttk.Label(stats_frame, text=strategy).grid(
                row=row, column=0, sticky=tk.W, pady=2)
            ttk.Label(stats_frame, text=str(count)).grid(
                row=row, column=1, sticky=tk.W, pady=2)
            row += 1
        
        # Add close button
        ttk.Button(stats_frame, text="Close", command=stats_window.destroy).grid(
            row=row, column=0, columnspan=2, pady=(20, 10))
    
    def visualize(self):
        """Generate and show visualizations"""
        try:
            # Generate the visualization file
            visual_file = self.ai.visualize_performance()
            
            # Show a message
            self.status_var.set(f"Visualization saved to {visual_file}")
            
            # Try to open the image in a new window
            visual_window = tk.Toplevel(self.root)
            visual_window.title("AI Performance Visualization")
            
            # Load the image
            from PIL import Image, ImageTk
            img = Image.open(visual_file)
            
            # Resize the image if it's too large
            screen_width = visual_window.winfo_screenwidth() - 100
            screen_height = visual_window.winfo_screenheight() - 100
            
            img_width, img_height = img.size
            if img_width > screen_width or img_height > screen_height:
                ratio = min(screen_width / img_width, screen_height / img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            
            # Display the image
            label = ttk.Label(visual_window, image=photo)
            label.image = photo  # Keep a reference to avoid garbage collection
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add close button
            ttk.Button(visual_window, text="Close", command=visual_window.destroy).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {e}")


def main():
    # Create the root window
    root = tk.Tk()
    
    # Set a theme (if ttkthemes is installed)
    try:
        from ttkthemes import ThemedStyle
        style = ThemedStyle(root)
        style.set_theme("arc")  # Other options: "equilux", "plastik", etc.
    except ImportError:
        pass
    
    # Create the app
    app = RockPaperScissorsGUI(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()
