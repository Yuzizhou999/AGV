"""
Circular Track Dual-Station Shut        self.ax_track.set_title('Track View', fontsize=14, fontweight='bold')
        
        # Right: Status information panel
        self.ax_info = plt.subplot(1, 2, 2)
        self.ax_info.axis('off')
        
        # History data recording
        self.history = {duling System - Visualization Module
Implements visualization of vehicle movement, cargo, and loading/unloading stations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PatchCollection
from typing import Dict, List, Optional
from config import *
from environment import Environment


class AGVVisualizer:
    """AGV System Visualizer"""
    
    def __init__(self, environment: Environment, save_path: Optional[str] = None):
        """
        Initialize visualizer
        
        Args:
            environment: Environment object
            save_path: Save path (optional, for saving animation)
        """
        self.env = environment
        self.save_path = save_path
        
        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('AGV Circular Track Scheduling System', fontsize=16, fontweight='bold')
        
        # Create subplots
        # Left: Circular track view
        self.ax_track = plt.subplot(1, 2, 1)
        self.ax_track.set_aspect('equal')
        self.ax_track.set_title('Track View', fontsize=14, fontweight='bold')
        
        # 右侧：状态信息面板
        self.ax_info = plt.subplot(1, 2, 2)
        self.ax_info.axis('off')
        
        # 历史数据记录
        self.history = {
            'time': [],
            'completed': [],
            'timeout': [],
            'avg_wait_time': [],
            'vehicle_positions': {i: [] for i in range(MAX_VEHICLES)},
            'vehicle_velocities': {i: [] for i in range(MAX_VEHICLES)},
            'cargo_count': []
        }
        
        # Initialize track view
        self._setup_track_view()
        
    def _setup_track_view(self):
        """Setup circular track view"""
        # Calculate circular track radius
        self.radius = TRACK_LENGTH / (2 * np.pi)
        
        # Draw track circle
        track_circle = plt.Circle((0, 0), self.radius, fill=False, 
                                 edgecolor='black', linewidth=3, linestyle='-')
        self.ax_track.add_patch(track_circle)
        
        # Inner circle (decoration)
        inner_circle = plt.Circle((0, 0), self.radius * 0.85, fill=False,
                                 edgecolor='gray', linewidth=1, linestyle='--', alpha=0.5)
        self.ax_track.add_patch(inner_circle)
        
        # Set axis range
        margin = self.radius * 0.3
        self.ax_track.set_xlim(-self.radius - margin, self.radius + margin)
        self.ax_track.set_ylim(-self.radius - margin, self.radius + margin)
        
        # Draw loading stations
        self.loading_station_patches = []
        for station_id, station in self.env.loading_stations.items():
            angle = self._position_to_angle(station.position)
            x, y = self._polar_to_cartesian(self.radius, angle)
            
            # Loading station marker (triangle, pointing to track inside)
            triangle = patches.FancyBboxPatch(
                (x - 0.8, y - 0.8), 1.6, 1.6,
                boxstyle="round,pad=0.1",
                edgecolor='green', facecolor='lightgreen',
                linewidth=2, alpha=0.7
            )
            self.ax_track.add_patch(triangle)
            self.loading_station_patches.append(triangle)
            
            # Label
            label_x = x * 1.2
            label_y = y * 1.2
            self.ax_track.text(label_x, label_y, f'Loading{station_id+1}',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Draw unloading stations
        self.unloading_station_patches = []
        for station_id, station in self.env.unloading_stations.items():
            angle = self._position_to_angle(station.position)
            x, y = self._polar_to_cartesian(self.radius, angle)
            
            # Unloading station marker (rectangle)
            rect = patches.FancyBboxPatch(
                (x - 0.8, y - 0.8), 1.6, 1.6,
                boxstyle="round,pad=0.1",
                edgecolor='blue', facecolor='lightblue',
                linewidth=2, alpha=0.7
            )
            self.ax_track.add_patch(rect)
            self.unloading_station_patches.append(rect)
            
            # Label
            label_x = x * 1.15
            label_y = y * 1.15
            self.ax_track.text(label_x, label_y, f'Unload{station_id+1}',
                             ha='center', va='center', fontsize=10,
                             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Initialize vehicle drawing objects (draw when updated)
        self.vehicle_patches = []
        self.vehicle_texts = []
        self.cargo_patches = []
        
    def _position_to_angle(self, position: float) -> float:
        """Convert track position to angle (radians)"""
        # position starts from 0, increases clockwise
        # angle starts from right (0 degrees), increases counterclockwise
        return -2 * np.pi * position / TRACK_LENGTH + np.pi / 2
    
    def _polar_to_cartesian(self, r: float, angle: float) -> tuple:
        """Polar to Cartesian coordinates"""
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return x, y
    
    def _update_vehicles(self):
        """Update vehicle display"""
        # Clear old vehicle graphics
        for patch in self.vehicle_patches:
            patch.remove()
        for text in self.vehicle_texts:
            text.remove()
        self.vehicle_patches.clear()
        self.vehicle_texts.clear()
        
        # Draw each vehicle
        colors = ['red', 'orange', 'purple', 'brown']
        for vehicle_id, vehicle in self.env.vehicles.items():
            angle = self._position_to_angle(vehicle.position)
            x, y = self._polar_to_cartesian(self.radius, angle)
            
            # Vehicle body (circle)
            color = colors[vehicle_id % len(colors)]
            vehicle_circle = plt.Circle((x, y), 1.2, 
                                       facecolor=color, edgecolor='darkred',
                                       linewidth=2, alpha=0.8, zorder=10)
            self.ax_track.add_patch(vehicle_circle)
            self.vehicle_patches.append(vehicle_circle)
            
            # Vehicle ID
            text = self.ax_track.text(x, y, f'V{vehicle_id}',
                                     ha='center', va='center',
                                     fontsize=11, fontweight='bold',
                                     color='white', zorder=11)
            self.vehicle_texts.append(text)
            
            # Velocity arrow (if has velocity)
            if abs(vehicle.velocity) > 0.1:
                # Calculate arrow direction (tangent direction)
                tangent_angle = angle - np.pi / 2  # Clockwise direction
                arrow_length = min(abs(vehicle.velocity) * 0.5, 3)
                dx = arrow_length * np.cos(tangent_angle)
                dy = arrow_length * np.sin(tangent_angle)
                
                arrow = patches.FancyArrowPatch(
                    (x, y), (x + dx, y + dy),
                    arrowstyle='->', mutation_scale=20,
                    color='yellow', linewidth=2, zorder=9
                )
                self.ax_track.add_patch(arrow)
                self.vehicle_patches.append(arrow)
            
            # Show cargo on slots
            occupied_slots = sum(1 for slot in vehicle.slots if slot is not None)
            if occupied_slots > 0:
                # Show cargo count next to vehicle
                cargo_text = self.ax_track.text(
                    x * 0.85, y * 0.85, f'Box x{occupied_slots}',
                    ha='center', va='center',
                    fontsize=9, bbox=dict(boxstyle='round', 
                                         facecolor='yellow', alpha=0.7),
                    zorder=11
                )
                self.vehicle_texts.append(cargo_text)
    
    def _update_cargos_on_stations(self):
        """Update cargo display on stations"""
        # Clear old cargo graphics
        for patch in self.cargo_patches:
            patch.remove()
        self.cargo_patches.clear()
        
        # Cargo at loading stations
        for station_id, station in self.env.loading_stations.items():
            angle = self._position_to_angle(station.position)
            x, y = self._polar_to_cartesian(self.radius, angle)
            
            for slot_idx, cargo_id in enumerate(station.slots):
                if cargo_id is not None:
                    # Cargo position (slightly offset)
                    offset = 0.4 if slot_idx == 0 else -0.4
                    cargo_x = x + offset * np.cos(angle + np.pi/2)
                    cargo_y = y + offset * np.sin(angle + np.pi/2)
                    
                    # Cargo marker (small square)
                    cargo_rect = patches.Rectangle(
                        (cargo_x - 0.2, cargo_y - 0.2), 0.4, 0.4,
                        facecolor='gold', edgecolor='orange',
                        linewidth=1, zorder=8
                    )
                    self.ax_track.add_patch(cargo_rect)
                    self.cargo_patches.append(cargo_rect)
        
        # Cargo at unloading stations
        for station_id, station in self.env.unloading_stations.items():
            angle = self._position_to_angle(station.position)
            x, y = self._polar_to_cartesian(self.radius, angle)
            
            for slot_idx, cargo_id in enumerate(station.slots):
                if cargo_id is not None:
                    # 货物位置（稍微偏移）
                    offset = 0.4 if slot_idx == 0 else -0.4
                    cargo_x = x + offset * np.cos(angle + np.pi/2)
                    cargo_y = y + offset * np.sin(angle + np.pi/2)
                    
                    # 货物标记（小方块，绿色表示已送达）
                    cargo_rect = patches.Rectangle(
                        (cargo_x - 0.2, cargo_y - 0.2), 0.4, 0.4,
                        facecolor='lightgreen', edgecolor='green',
                        linewidth=1, zorder=8
                    )
                    self.ax_track.add_patch(cargo_rect)
                    self.cargo_patches.append(cargo_rect)
    
    def _update_info_panel(self):
        """Update status information panel"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        # Title
        info_text = f"System Status (Time: {self.env.current_time:.1f}s)\n"
        info_text += "=" * 50 + "\n\n"
        
        # Statistics
        info_text += "Task Statistics:\n"
        info_text += f"  Completed: {self.env.completed_cargos}\n"
        info_text += f"  Timeout: {self.env.timed_out_cargos}\n"
        if self.env.completed_cargos > 0:
            avg_wait = self.env.total_wait_time / self.env.completed_cargos
            info_text += f"  Avg Wait Time: {avg_wait:.1f}s\n"
        info_text += f"  Current Cargos: {len(self.env.cargos)}\n\n"
        
        # Vehicle status
        info_text += "Vehicle Status:\n"
        for vehicle_id, vehicle in self.env.vehicles.items():
            info_text += f"  Vehicle {vehicle_id}:\n"
            info_text += f"    Position: {vehicle.position:.1f}\n"
            info_text += f"    Velocity: {vehicle.velocity:.2f}\n"
            
            slot_info = []
            for i, cargo_id in enumerate(vehicle.slots):
                if cargo_id is not None:
                    cargo = self.env.cargos.get(cargo_id)
                    if cargo:
                        slot_info.append(f"Slot{i+1}: Cargo{cargo_id}")
                else:
                    slot_info.append(f"Slot{i+1}: Empty")
            info_text += f"    Slots: {' | '.join(slot_info)}\n"
            
            if vehicle.is_loading_unloading:
                info_text += f"    Status: Loading/Unloading\n"
            else:
                info_text += f"    Status: Ready\n"
            info_text += "\n"
        
        # Loading station status
        info_text += "Loading Station Status:\n"
        for station_id, station in self.env.loading_stations.items():
            info_text += f"  Loading {station_id} (Pos {station.position:.1f}):\n"
            for slot_idx, cargo_id in enumerate(station.slots):
                if cargo_id is not None:
                    cargo = self.env.cargos.get(cargo_id)
                    if cargo:
                        wait_time = cargo.wait_time(self.env.current_time)
                        timeout_marker = "WARNING" if cargo.is_timeout(self.env.current_time) else ""
                        info_text += f"    Slot{slot_idx+1}: Cargo{cargo_id} (Wait{wait_time:.1f}s) {timeout_marker}\n"
                else:
                    info_text += f"    Slot{slot_idx+1}: Empty\n"
        info_text += "\n"
        
        # Unloading station status
        info_text += "Unloading Station Status:\n"
        for station_id, station in self.env.unloading_stations.items():
            info_text += f"  Unloading {station_id} (Pos {station.position:.1f}):\n"
            for slot_idx, cargo_id in enumerate(station.slots):
                if cargo_id is not None:
                    info_text += f"    Slot{slot_idx+1}: Cargo{cargo_id} OK\n"
                else:
                    status = "Reserved" if station.slot_reserved[slot_idx] else "Empty"
                    info_text += f"    Slot{slot_idx+1}: {status}\n"
        info_text += "\n"
        
        # Completed cargo details (show last 5)
        if hasattr(self.env, 'completed_cargo_list') and len(self.env.completed_cargo_list) > 0:
            info_text += "Recently Completed Cargos:\n"
            recent_completed = self.env.completed_cargo_list[-5:]  # Last 5
            for cargo_info in recent_completed:
                info_text += f"  Cargo{cargo_info['id']}: "
                info_text += f"Load{cargo_info['loading_station']}->"
                info_text += f"Unload{cargo_info['unloading_station']} "
                info_text += f"(Wait: {cargo_info['wait_time']:.1f}s, "
                info_text += f"Vehicle: {cargo_info['vehicle_id']})\n"
        
        # Display text
        self.ax_info.text(0.05, 0.95, info_text,
                         transform=self.ax_info.transAxes,
                         fontsize=10, family='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def update(self):
        """Update visualization"""
        self._update_vehicles()
        self._update_cargos_on_stations()
        self._update_info_panel()
        
        # Record history data
        self.history['time'].append(self.env.current_time)
        self.history['completed'].append(self.env.completed_cargos)
        self.history['timeout'].append(self.env.timed_out_cargos)
        if self.env.completed_cargos > 0:
            avg_wait = self.env.total_wait_time / self.env.completed_cargos
            self.history['avg_wait_time'].append(avg_wait)
        else:
            self.history['avg_wait_time'].append(0)
        self.history['cargo_count'].append(len(self.env.cargos))
        
        for vehicle_id, vehicle in self.env.vehicles.items():
            self.history['vehicle_positions'][vehicle_id].append(vehicle.position)
            self.history['vehicle_velocities'][vehicle_id].append(vehicle.velocity)
        
        plt.draw()
        plt.pause(0.001)
    
    def show(self):
        """Show visualization window"""
        plt.show()
    
    def close(self):
        """Close visualization window"""
        plt.close(self.fig)
    
    def plot_statistics(self, save_path: Optional[str] = None):
        """Plot statistics charts"""
        if len(self.history['time']) == 0:
            print("No history data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('System Performance Statistics', fontsize=16, fontweight='bold')
        
        time = np.array(self.history['time'])
        
        # 1. Completed and timeout cargo count
        ax1 = axes[0, 0]
        ax1.plot(time, self.history['completed'], label='Completed', color='green', linewidth=2)
        ax1.plot(time, self.history['timeout'], label='Timeout', color='red', linewidth=2)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cargo Count')
        ax1.set_title('Cargo Completion and Timeout')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Average wait time
        ax2 = axes[0, 1]
        ax2.plot(time, self.history['avg_wait_time'], color='blue', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Avg Wait Time (s)')
        ax2.set_title('Average Wait Time Trend')
        ax2.grid(True, alpha=0.3)
        
        # 3. Vehicle position
        ax3 = axes[1, 0]
        colors = ['red', 'orange', 'purple', 'brown']
        for vehicle_id in range(MAX_VEHICLES):
            positions = self.history['vehicle_positions'][vehicle_id]
            ax3.plot(time, positions, label=f'Vehicle {vehicle_id}',
                    color=colors[vehicle_id % len(colors)], linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position')
        ax3.set_title('Vehicle Position Trajectory')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Current cargo count
        ax4 = axes[1, 1]
        ax4.plot(time, self.history['cargo_count'], color='purple', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Cargo Count')
        ax4.set_title('System Cargo Count Trend')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistics chart saved to: {save_path}")
        
        plt.show()


def visualize_training(env: Environment, update_interval: float = 0.1):
    """
    Real-time visualization of training process
    
    Args:
        env: Environment object
        update_interval: Visualization update interval (seconds)
    """
    visualizer = AGVVisualizer(env)
    visualizer.update()
    return visualizer


if __name__ == "__main__":
    # Test visualization
    from environment import Environment
    
    env = Environment(seed=42)
    visualizer = AGVVisualizer(env)
    
    # Simulate several steps
    for _ in range(100):
        # Simply move vehicles
        for vehicle in env.vehicles.values():
            vehicle.position = (vehicle.position + 0.5) % TRACK_LENGTH
            vehicle.velocity = 2.0
        
        # Update time
        env.current_time += 1.0
        
        # Update visualization
        visualizer.update()
    
    # Show statistics
    visualizer.plot_statistics()
    visualizer.show()
