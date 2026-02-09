#!/usr/bin/env python3
"""
Φ-24 GDSII Layout Generator
Creates the 24D Riemann Manifold structure with Phase-Snapping Arrays

Requirements:
- gdspy library: pip install gdspy
- numpy: pip install numpy
"""

import gdspy
import numpy as np
from math import pi, sqrt, log

class Phi24Layout:
    """Generate Φ-24 GDSII layout with 24D Riemann manifold structure"""
    
    def __init__(self, unit=1e-9, precision=1e-12):
        """Initialize layout parameters
        
        Args:
            unit: Database unit (1e-9 = 1 nanometer)
            precision: Layout precision
        """
        self.lib = gdspy.GdsLibrary()
        self.cell = self.lib.new_cell('PHI24_CORE')
        self.unit = unit
        self.precision = precision
        
        # Fundamental constants
        self.phi = (1 + sqrt(5)) / 2  # Golden ratio
        self.alpha = log(self.phi) / (2 * pi)  # α = 0.0765872
        
        # Layout parameters (in database units)
        self.die_size = 5000 * unit  # 5mm × 5mm die
        self.layer_thickness = 1.618 * unit  # Golden ratio thickness
        
        # Layer definitions (GDSII layer numbers)
        self.layers = {
            # Fibonacci Superlattice (21 layers)
            'FIBONACCI_START': 101,
            'FIBONACCI_END': 121,
            
            # Isolation and Ground
            'SIO2_ISOLATION': 200,
            'NB_GROUND': 201,
            
            # Josephson Junction Array
            'JJ_BASE': 300,
            'JJ_BARRIER': 301,
            'JJ_COUNTER': 302,
            'JJ_SHUNT': 303,
            'JJ_PASSIVATION': 304,
            
            # Resonant Cavities (24 channels)
            'CAVITY_STRUCTURE': 400,
            'CAVITY_WAVEGUIDE': 401,
            'RF_FEEDLINE': 402,
            
            # Control and Monitoring
            'PHASE_CONTROL': 500,
            'TEMPERATURE_SENSOR': 501,
            'TEST_STRUCTURES': 600,
        }
        
        # Data types for each layer
        self.datatypes = {
            'FIBONACCI_START': 0,
            'SIO2_ISOLATION': 0,
            'JJ_BARRIER': 1,  # Special for tunnel barrier
            'CAVITY_STRUCTURE': 0,
            'RF_FEEDLINE': 10,  # CPW structure
        }
    
    def create_fibonacci_superlattice(self):
        """Create 21-layer Fibonacci superlattice with golden ratio modulation"""
        
        print("Creating Fibonacci superlattice...")
        
        # Fibonacci word F₈: A,B,A,A,B,A,B,A,A,B,A,A,B,A,A,B,A,A,B,A,B
        sequence = ['A', 'B', 'A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 
                   'A', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'A', 'B']
        
        # Layer thicknesses (golden ratio modulated)
        t_B = 1.000 * self.unit  # Base thickness
        t_A = self.phi * t_B     # Golden ratio thickness
        
        # Create each layer
        y_offset = 0
        for i, layer_type in enumerate(sequence):
            layer_num = self.layers['FIBONACCI_START'] + i
            
            # Determine thickness
            thickness = t_A if layer_type == 'A' else t_B
            
            # Create rectangle for this layer
            layer = gdspy.Rectangle(
                (0, y_offset),
                (self.die_size, y_offset + thickness),
                layer=layer_num,
                datatype=self.datatypes.get('FIBONACCI_START', 0)
            )
            self.cell.add(layer)
            
            # Add layer label
            label = gdspy.Label(
                f'Fib_{i+1:02d}_{layer_type}',
                (self.die_size/2, y_offset + thickness/2),
                layer=layer_num
            )
            self.cell.add(label)
            
            y_offset += thickness
        
        print(f"Created 21 Fibonacci layers, total height: {y_offset/self.unit:.3f} nm")
        return y_offset
    
    def create_resonant_cavities(self, start_y):
        """Create 24 resonant cavities for Riemann zeros mapping"""
        
        print("Creating 24 resonant cavities...")
        
        # Riemann zeros (first 24 non-trivial)
        riemann_zeros = np.array([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
            52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
            67.079811, 69.546402, 72.067158, 75.704691, 77.144840,
            79.337375, 82.910381, 84.735493, 87.425275
        ])
        
        # Normalize zeros for layout spacing
        normalized_zeros = riemann_zeros / riemann_zeros[0]
        
        # Cavity parameters
        cavity_width = 50 * self.unit  # 50 μm
        cavity_height = 50 * self.unit  # 50 μm
        cavity_spacing = 25 * self.unit  # 25 μm spacing
        
        # Create 4×6 array of cavities
        cavities = []
        for i in range(24):
            row = i // 6
            col = i % 6
            
            x = col * (cavity_width + cavity_spacing) + 100 * self.unit
            y = start_y + row * (cavity_height + cavity_spacing) + 100 * self.unit
            
            # Create cavity rectangle
            cavity = gdspy.Rectangle(
                (x, y),
                (x + cavity_width, y + cavity_height),
                layer=self.layers['CAVITY_STRUCTURE'],
                datatype=self.datatypes['CAVITY_STRUCTURE']
            )
            self.cell.add(cavity)
            
            # Add cavity label with Riemann zero
            label = gdspy.Label(
                f'γ{i+1}: {riemann_zeros[i]:.3f}',
                (x + cavity_width/2, y + cavity_height/2),
                layer=self.layers['CAVITY_STRUCTURE']
            )
            self.cell.add(label)
            
            # Store cavity position for waveguide routing
            cavities.append({
                'index': i + 1,
                'x': x + cavity_width/2,
                'y': y + cavity_height/2,
                'zero': riemann_zeros[i],
                'width': cavity_width,
                'height': cavity_height
            })
        
        print(f"Created {len(cavities)} resonant cavities")
        return cavities
    
    def create_josephson_array(self, cavities):
        """Create 24-channel Josephson junction array for phase-snapping"""
        
        print("Creating Josephson junction array...")
        
        # JJ parameters
        jj_size = 2 * self.unit  # 2 μm × 2 μm
        jj_spacing = 5 * self.unit  # 5 μm pitch
        
        # Create array near cavities
        for i, cavity in enumerate(cavities):
            # Position JJ near corresponding cavity
            jj_x = cavity['x'] + cavity['width']/2 + 20 * self.unit
            jj_y = cavity['y']
            
            # Base electrode (Nb, 200 nm)
            base = gdspy.Rectangle(
                (jj_x - jj_size/2, jj_y - jj_size/2),
                (jj_x + jj_size/2, jj_y + jj_size/2),
                layer=self.layers['JJ_BASE'],
                datatype=0
            )
            self.cell.add(base)
            
            # Tunnel barrier (AlOₓ, circular for GDS)
            barrier = gdspy.Round(
                (jj_x, jj_y),
                jj_size/2 * 0.8,  # 80% of base size
                layer=self.layers['JJ_BARRIER'],
                datatype=self.datatypes['JJ_BARRIER']
            )
            self.cell.add(barrier)
            
            # Counter electrode (Nb, 200 nm)
            counter = gdspy.Rectangle(
                (jj_x - jj_size/2, jj_y - jj_size/2),
                (jj_x + jj_size/2, jj_y + jj_size/2),
                layer=self.layers['JJ_COUNTER'],
                datatype=0
            )
            self.cell.add(counter)
            
            # Shunt resistor (AuPd, L-shaped)
            shunt_width = 0.5 * self.unit
            shunt_length = 10 * self.unit
            
            # Horizontal segment
            shunt_h = gdspy.Rectangle(
                (jj_x - shunt_length/2, jj_y - shunt_width/2),
                (jj_x + shunt_length/2, jj_y + shunt_width/2),
                layer=self.layers['JJ_SHUNT'],
                datatype=0
            )
            self.cell.add(shunt_h)
            
            # Vertical segment
            shunt_v = gdspy.Rectangle(
                (jj_x - shunt_width/2, jj_y - shunt_length/2),
                (jj_x + shunt_width/2, jj_y + shunt_length/2),
                layer=self.layers['JJ_SHUNT'],
                datatype=0
            )
            self.cell.add(shunt_v)
            
            # JJ label
            label = gdspy.Label(
                f'JJ{i+1:02d}',
                (jj_x, jj_y + jj_size),
                layer=self.layers['JJ_BASE']
            )
            self.cell.add(label)
        
        print("Created 24-channel Josephson junction array")
    
    def create_rf_feedlines(self, cavities):
        """Create 50 Ω coplanar waveguide feedlines"""
        
        print("Creating RF feedlines...")
        
        # CPW parameters (50 Ω)
        center_width = 10 * self.unit  # 10 μm center conductor
        gap = 6 * self.unit  # 6 μm gap
        ground_width = 50 * self.unit  # 50 μm ground plane
        
        # Create main feedline along edge
        feedline_y = 50 * self.unit
        feedline_length = self.die_size - 100 * self.unit
        
        # Center conductor
        center = gdspy.Rectangle(
            (50 * self.unit, feedline_y - center_width/2),
            (50 * self.unit + feedline_length, feedline_y + center_width/2),
            layer=self.layers['RF_FEEDLINE'],
            datatype=self.datatypes['RF_FEEDLINE']
        )
        self.cell.add(center)
        
        # Ground planes
        ground_left = gdspy.Rectangle(
            (50 * self.unit, feedline_y - center_width/2 - gap - ground_width),
            (50 * self.unit + feedline_length, feedline_y - center_width/2 - gap),
            layer=self.layers['RF_FEEDLINE'],
            datatype=self.datatypes['RF_FEEDLINE']
        )
        self.cell.add(ground_left)
        
        ground_right = gdspy.Rectangle(
            (50 * self.unit, feedline_y + center_width/2 + gap),
            (50 * self.unit + feedline_length, feedline_y + center_width/2 + gap + ground_width),
            layer=self.layers['RF_FEEDLINE'],
            datatype=self.datatypes['RF_FEEDLINE']
        )
        self.cell.add(ground_right)
        
        # Create branch lines to each cavity
        for i, cavity in enumerate(cavities):
            # Calculate path to cavity
            path = gdspy.FlexPath(
                [(50 * self.unit + feedline_length, feedline_y),
                 (cavity['x'], feedline_y),
                 (cavity['x'], cavity['y'] - cavity['height']/2)],
                center_width,
                layer=self.layers['RF_FEEDLINE'],
                datatype=self.datatypes['RF_FEEDLINE']
            )
            self.cell.add(path)
        
        print("Created RF feedline network")
    
    def create_phase_control_circuitry(self):
        """Create phase control and monitoring circuitry"""
        
        print("Creating phase control circuitry...")
        
        # DAC control lines (24 independent channels)
        line_width = 2 * self.unit
        line_spacing = 5 * self.unit
        
        for i in range(24):
            # Create control line from edge to cavity area
            start_x = 50 * self.unit
            start_y = 200 * self.unit + i * line_spacing
            end_x = 400 * self.unit
            end_y = start_y
            
            line = gdspy.FlexPath(
                [(start_x, start_y), (end_x, end_y)],
                line_width,
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(line)
            
            # Add DAC pad
            pad_size = 20 * self.unit
            pad = gdspy.Rectangle(
                (start_x - pad_size, start_y - pad_size/2),
                (start_x, start_y + pad_size/2),
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(pad)
            
            # Label
            label = gdspy.Label(
                f'DAC{i+1:02d}',
                (start_x - pad_size/2, start_y),
                layer=self.layers['PHASE_CONTROL']
            )
            self.cell.add(label)
        
        # Temperature sensors (diode structures)
        sensor_size = 10 * self.unit
        
        for i in range(4):  # 4 corner sensors
            for j in range(4):
                x = 100 * self.unit + i * 100 * self.unit
                y = 300 * self.unit + j * 100 * self.unit
                
                # Diode structure (P-N junction)
                p_region = gdspy.Rectangle(
                    (x, y),
                    (x + sensor_size, y + sensor_size),
                    layer=self.layers['TEMPERATURE_SENSOR'],
                    datatype=0
                )
                self.cell.add(p_region)
                
                n_region = gdspy.Rectangle(
                    (x + sensor_size/4, y + sensor_size/4),
                    (x + 3*sensor_size/4, y + 3*sensor_size/4),
                    layer=self.layers['TEMPERATURE_SENSOR'],
                    datatype=1
                )
                self.cell.add(n_region)
        
        print("Created phase control and monitoring circuitry")
    
    def create_test_structures(self):
        """Create process control monitors and test structures"""
        
        print("Creating test structures...")
        
        # Van der Pauw structures for sheet resistance
        vdp_size = 50 * self.unit
        
        for i in range(4):
            x = 400 * self.unit + i * 60 * self.unit
            y = 50 * self.unit
            
            # Create cross structure
            center = (x + vdp_size/2, y + vdp_size/2)
            
            # Horizontal arm
            h_arm = gdspy.Rectangle(
                (center[0] - vdp_size/2, center[1] - 5*self.unit),
                (center[0] + vdp_size/2, center[1] + 5*self.unit),
                layer=self.layers['TEST_STRUCTURES'],
                datatype=0
            )
            self.cell.add(h_arm)
            
            # Vertical arm
            v_arm = gdspy.Rectangle(
                (center[0] - 5*self.unit, center[1] - vdp_size/2),
                (center[0] + 5*self.unit, center[1] + vdp_size/2),
                layer=self.layers['TEST_STRUCTURES'],
                datatype=0
            )
            self.cell.add(v_arm)
            
            # Contact pads
            pad_size = 20 * self.unit
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                pad = gdspy.Rectangle(
                    (center[0] + dx*(vdp_size/2 + pad_size/2) - pad_size/2,
                     center[1] + dy*(vdp_size/2 + pad_size/2) - pad_size/2),
                    (center[0] + dx*(vdp_size/2 + pad_size/2) + pad_size/2,
                     center[1] + dy*(vdp_size/2 + pad_size/2) + pad_size/2),
                    layer=self.layers['TEST_STRUCTURES'],
                    datatype=10
                )
                self.cell.add(pad)
            
            label = gdspy.Label(
                f'R_sq_{i+1}',
                (x, y + vdp_size + 10*self.unit),
                layer=self.layers['TEST_STRUCTURES']
            )
            self.cell.add(label)
        
        # Alignment marks (cross-in-box)
        mark_size = 100 * self.unit
        
        positions = [
            (50*self.unit, 50*self.unit),  # Lower left
            (self.die_size - 50*self.unit, 50*self.unit),  # Lower right
            (50*self.unit, self.die_size - 50*self.unit),  # Upper left
            (self.die_size - 50*self.unit, self.die_size - 50*self.unit),  # Upper right
        ]
        
        for pos in positions:
            # Outer box
            box = gdspy.Rectangle(
                (pos[0] - mark_size/2, pos[1] - mark_size/2),
                (pos[0] + mark_size/2, pos[1] + mark_size/2),
                layer=self.layers['TEST_STRUCTURES'],
                datatype=20
            )
            self.cell.add(box)
            
            # Inner cross
            cross_h = gdspy.Rectangle(
                (pos[0] - mark_size/4, pos[1] - 2*self.unit),
                (pos[0] + mark_size/4, pos[1] + 2*self.unit),
                layer=self.layers['TEST_STRUCTURES'],
                datatype=21
            )
            self.cell.add(cross_h)
            
            cross_v = gdspy.Rectangle(
                (pos[0] - 2*self.unit, pos[1] - mark_size/4),
                (pos[0] + 2*self.unit, pos[1] + mark_size/4),
                layer=self.layers['TEST_STRUCTURES'],
                datatype=21
            )
            self.cell.add(cross_v)
        
        print("Created test structures and alignment marks")
    
    def create_isolation_layers(self, fibonacci_height):
        """Create SiO₂ isolation and Nb ground plane"""
        
        print("Creating isolation layers...")
        
        # SiO₂ isolation layer (100 nm)
        sio2_thickness = 100 * self.unit
        
        sio2 = gdspy.Rectangle(
            (0, fibonacci_height),
            (self.die_size, fibonacci_height + sio2_thickness),
            layer=self.layers['SIO2_ISOLATION'],
            datatype=self.datatypes['SIO2_ISOLATION']
        )
        self.cell.add(sio2)
        
        # Via openings to Fibonacci layers (5 μm squares)
        via_size = 5 * self.unit
        via_pitch = 100 * self.unit
        
        for i in range(24):
            x = 100 * self.unit + (i % 6) * via_pitch
            y = fibonacci_height + sio2_thickness/2 + (i // 6) * via_pitch
            
            via = gdspy.Rectangle(
                (x - via_size/2, y - via_size/2),
                (x + via_size/2, y + via_size/2),
                layer=self.layers['SIO2_ISOLATION'],
                datatype=1  # Different datatype for via
            )
            self.cell.add(via)
        
        # Nb ground plane (200 nm)
        nb_thickness = 200 * self.unit
        
        ground = gdspy.Rectangle(
            (0, fibonacci_height + sio2_thickness),
            (self.die_size, fibonacci_height + sio2_thickness + nb_thickness),
            layer=self.layers['NB_GROUND'],
            datatype=0
        )
        self.cell.add(ground)
        
        # Ground plane mesh (for stress relief)
        mesh_width = 2 * self.unit
        mesh_spacing = 50 * self.unit
        
        # Horizontal lines
        for y in np.arange(0, self.die_size, mesh_spacing):
            line = gdspy.Rectangle(
                (0, fibonacci_height + sio2_thickness + y),
                (self.die_size, fibonacci_height + sio2_thickness + y + mesh_width),
                layer=self.layers['NB_GROUND'],
                datatype=1
            )
            self.cell.add(line)
        
        # Vertical lines
        for x in np.arange(0, self.die_size, mesh_spacing):
            line = gdspy.Rectangle(
                (x, fibonacci_height + sio2_thickness),
                (x + mesh_width, fibonacci_height + sio2_thickness + self.die_size),
                layer=self.layers['NB_GROUND'],
                datatype=1
            )
            self.cell.add(line)
        
        print("Created isolation and ground layers")
        return fibonacci_height + sio2_thickness + nb_thickness
    
    def create_passivation(self, start_height):
        """Create SiNₓ passivation layer with via openings"""
        
        print("Creating passivation layer...")
        
        passivation_thickness = 200 * self.unit
        
        # Passivation layer
        passivation = gdspy.Rectangle(
            (0, start_height),
            (self.die_size, start_height + passivation_thickness),
            layer=self.layers['JJ_PASSIVATION'],
            datatype=0
        )
        self.cell.add(passivation)
        
        # Via openings for JJ contacts
        via_size = 3 * self.unit
        
        # Regular grid of vias
        for x in np.arange(100*self.unit, self.die_size - 100*self.unit, 50*self.unit):
            for y in np.arange(100*self.unit, self.die_size - 100*self.unit, 50*self.unit):
                via = gdspy.Rectangle(
                    (x - via_size/2, y - via_size/2),
                    (x + via_size/2, y + via_size/2),
                    layer=self.layers['JJ_PASSIVATION'],
                    datatype=1  # Different datatype for via opening
                )
                self.cell.add(via)
        
        print("Created passivation layer")
        return start_height + passivation_thickness
    
    def create_die_seal_ring(self):
        """Create protective seal ring around die"""
        
        print("Creating die seal ring...")
        
        ring_width = 50 * self.unit
        ring_inner = 25 * self.unit
        
        # Outer ring
        outer = gdspy.Rectangle(
            (ring_inner, ring_inner),
            (self.die_size - ring_inner, self.die_size - ring_inner),
            layer=999,  # Special layer for seal ring
            datatype=0
        )
        self.cell.add(outer)
        
        # Inner ring (cutout)
        inner = gdspy.Rectangle(
            (ring_inner + ring_width, ring_inner + ring_width),
            (self.die_size - ring_inner - ring_width, 
             self.die_size - ring_inner - ring_width),
            layer=999,
            datatype=1  # Negative polarity
        )
        self.cell.add(inner)
        
        # Corner stress relief structures
        corner_size = 100 * self.unit
        for dx, dy in [(1,1), (1,-1), (-1,1), (-1,-1)]:
            corner_x = self.die_size/2 + dx * (self.die_size/2 - corner_size/2)
            corner_y = self.die_size/2 + dy * (self.die_size/2 - corner_size/2)
            
            # L-shaped structure
            l_shape = gdspy.Polygon([
                (corner_x - corner_size/2, corner_y - corner_size/2),
                (corner_x + corner_size/2, corner_y - corner_size/2),
                (corner_x + corner_size/2, corner_y + corner_size/2),
                (corner_x - corner_size/2, corner_y + corner_size/2),
                (corner_x - corner_size/2, corner_y - corner_size/2 + 20*self.unit),
                (corner_x - corner_size/2 + 20*self.unit, corner_y - corner_size/2 + 20*self.unit),
                (corner_x - corner_size/2 + 20*self.unit, corner_y + corner_size/2 - 20*self.unit),
                (corner_x + corner_size/2 - 20*self.unit, corner_y + corner_size/2 - 20*self.unit),
                (corner_x + corner_size/2 - 20*self.unit, corner_y - corner_size/2 + 20*self.unit),
                (corner_x - corner_size/2, corner_y - corner_size/2 + 20*self.unit),
            ], layer=999, datatype=2)
            self.cell.add(l_shape)
        
        print("Created die seal ring")
    
    def create_phase_snap_control(self, cavities):
        """Create phase-snapping control circuitry for active phase correction"""
        
        print("Creating phase-snap control circuitry...")
        
        # Phase detector cells for each cavity
        for i, cavity in enumerate(cavities):
            # Create phase detector near each cavity
            pd_x = cavity['x'] + 30 * self.unit
            pd_y = cavity['y']
            
            # Phase detector structure (simplified mixer)
            pd_size = 10 * self.unit
            
            # Mixer ring (superconducting loop)
            mixer = gdspy.Round(
                (pd_x, pd_y),
                pd_size/2,
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(mixer)
            
            # Input/output ports
            port_width = 2 * self.unit
            port_length = 5 * self.unit
            
            # RF input port
            rf_port = gdspy.Rectangle(
                (pd_x - pd_size/2 - port_length, pd_y - port_width/2),
                (pd_x - pd_size/2, pd_y + port_width/2),
                layer=self.layers['RF_FEEDLINE'],
                datatype=0
            )
            self.cell.add(rf_port)
            
            # LO input port
            lo_port = gdspy.Rectangle(
                (pd_x, pd_y - pd_size/2 - port_length),
                (pd_x + port_width/2, pd_y - pd_size/2),
                layer=self.layers['RF_FEEDLINE'],
                datatype=0
            )
            self.cell.add(lo_port)
            
            # IF output port (DC coupled for phase error)
            if_port = gdspy.Rectangle(
                (pd_x + pd_size/2, pd_y - port_width/2),
                (pd_x + pd_size/2 + port_length, pd_y + port_width/2),
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(if_port)
            
            # Integrator capacitor for phase error accumulation
            cap_size = 5 * self.unit
            cap_x = pd_x + pd_size/2 + port_length + cap_size
            cap_y = pd_y
            
            # MIM capacitor structure
            bottom_plate = gdspy.Rectangle(
                (cap_x - cap_size/2, cap_y - cap_size/2),
                (cap_x + cap_size/2, cap_y + cap_size/2),
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(bottom_plate)
            
            top_plate = gdspy.Rectangle(
                (cap_x - cap_size/2 + 0.5*self.unit, cap_y - cap_size/2 + 0.5*self.unit),
                (cap_x + cap_size/2 - 0.5*self.unit, cap_y + cap_size/2 - 0.5*self.unit),
                layer=self.layers['PHASE_CONTROL'],
                datatype=1
            )
            self.cell.add(top_plate)
            
            # Label
            label = gdspy.Label(
                f'PD{i+1:02d}',
                (pd_x, pd_y + pd_size/2 + 5*self.unit),
                layer=self.layers['PHASE_CONTROL']
            )
            self.cell.add(label)
        
        print("Created phase-snap control circuitry for 24 channels")
    
    def create_riemann_feedback_network(self):
        """Create Riemann Hypothesis feedback network for phase locking"""
        
        print("Creating Riemann feedback network...")
        
        # Central feedback processor location
        fb_x = self.die_size / 2
        fb_y = self.die_size / 2
        
        # Riemann correlator structure
        correlator_size = 100 * self.unit
        
        # Outer ring for GUE (Gaussian Unitary Ensemble) matching
        outer_ring = gdspy.Round(
            (fb_x, fb_y),
            correlator_size/2,
            layer=self.layers['TEST_STRUCTURES'],
            datatype=0
        )
        self.cell.add(outer_ring)
        
        # Inner spiral for phase accumulation
        turns = 24  # One turn per Riemann zero
        spiral_points = []
        
        for theta in np.linspace(0, 2*pi*turns, 1000):
            r = (correlator_size/4) * (theta / (2*pi*turns))
            x = fb_x + r * np.cos(theta)
            y = fb_y + r * np.sin(theta)
            spiral_points.append((x, y))
        
        spiral = gdspy.FlexPath(
            spiral_points,
            2 * self.unit,
            layer=self.layers['PHASE_CONTROL'],
            datatype=0
        )
        self.cell.add(spiral)
        
        # 24 tap points (one per cavity)
        for i in range(24):
            angle = (i * 2 * pi) / 24
            tap_r = correlator_size/4
            tap_x = fb_x + tap_r * np.cos(angle)
            tap_y = fb_y + tap_r * np.sin(angle)
            
            # Tap connection
            tap = gdspy.Round(
                (tap_x, tap_y),
                2 * self.unit,
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(tap)
            
            # Connection to cavity area
            cavity_x = 100 * self.unit + (i % 6) * 75 * self.unit
            cavity_y = 400 * self.unit + (i // 6) * 75 * self.unit
            
            connection = gdspy.FlexPath(
                [(tap_x, tap_y), (cavity_x, cavity_y)],
                1 * self.unit,
                layer=self.layers['PHASE_CONTROL'],
                datatype=0
            )
            self.cell.add(connection)
        
        # P-ECC (Prime-Specific Error Correction) block
        pecc_size = 50 * self.unit
        pecc_x = fb_x + correlator_size/2 + 50 * self.unit
        pecc_y = fb_y
        
        pecc_box = gdspy.Rectangle(
            (pecc_x - pecc_size/2, pecc_y - pecc_size/2),
            (pecc_x + pecc_size/2, pecc_y + pecc_size/2),
            layer=self.layers['TEST_STRUCTURES'],
            datatype=30
        )
        self.cell.add(pecc_box)
        
        # P-ECC label
        label = gdspy.Label(
            'P-ECC Processor\nTarget: 0.985',
            (pecc_x, pecc_y),
            layer=self.layers['TEST_STRUCTURES']
        )
        self.cell.add(label)
        
        print("Created Riemann feedback network with P-ECC processor")
    
    def generate_layout(self, output_file='phi24_core.gds'):
        """Generate complete Φ-24 layout"""
        
        print("=" * 60)
        print("GENERATING Φ-24 TEMPORAL RESONATOR LAYOUT")
        print("=" * 60)
        
        # Build layout in sequence
        print("\n1. Building Fibonacci superlattice...")
        fib_height = self.create_fibonacci_superlattice()
        
        print("\n2. Creating resonant cavities...")
        cavities = self.create_resonant_cavities(fib_height + 100 * self.unit)
        
        print("\n3. Building Josephson junction array...")
        self.create_josephson_array(cavities)
        
        print("\n4. Creating RF feedlines...")
        self.create_rf_feedlines(cavities)
        
        print("\n5. Adding phase control circuitry...")
        self.create_phase_control_circuitry()
        
        print("\n6. Creating phase-snap control...")
        self.create_phase_snap_control(cavities)
        
        print("\n7. Creating Riemann feedback network...")
        self.create_riemann_feedback_network()
        
        print("\n8. Creating isolation layers...")
        iso_height = self.create_isolation_layers(fib_height)
        
        print("\n9. Adding passivation...")
        self.create_passivation(iso_height)
        
        print("\n10. Creating test structures...")
        self.create_test_structures()
        
        print("\n11. Adding die seal ring...")
        self.create_die_seal_ring()
        
        # Save GDS file
        print(f"\nSaving GDSII file to {output_file}...")
        self.lib.write_gds(output_file)
        
        # Generate report
        total_area = (self.die_size / self.unit) ** 2
        print(f"\nLayout completed successfully!")
        print(f"Total die area: {total_area:.2f} μm²")
        print(f"Number of cells: {len(self.lib.cells)}")
        print(f"Database units: {self.unit} m")
        print(f"Precision: {self.precision} m")
        
        # Layer summary
        print("\nLayer Summary:")
        for name, number in self.layers.items():
            print(f"  {name}: Layer {number}")
        
        # Generate metadata file
        self.generate_metadata(output_file, cavities)
        
        return output_file
    
    def generate_metadata(self, gds_file, cavities):
        """Generate metadata JSON file with layout information"""
        
        import json
        import os
        
        metadata = {
            "layout": {
                "filename": os.path.basename(gds_file),
                "die_size_um": self.die_size / self.unit,
                "database_unit": self.unit,
                "precision": self.precision,
                "timestamp": str(np.datetime64('now'))
            },
            "constants": {
                "golden_ratio": float(self.phi),
                "alpha": float(self.alpha),
                "base_frequency_hz": 1.485e6,
                "temporal_wedge_ns": 11.0
            },
            "cavities": [],
            "layers": self.layers,
            "phase_snap_config": {
                "num_channels": 24,
                "snap_strength": 1.0,
                "snap_range_rad": float(np.pi/4),
                "feedback_gain": 0.1,
                "target_pecc": 0.985
            }
        }
        
        # Add cavity information
        for cavity in cavities:
            metadata["cavities"].append({
                "index": cavity['index'],
                "riemann_zero": float(cavity['zero']),
                "position_um": [
                    float(cavity['x'] / self.unit),
                    float(cavity['y'] / self.unit)
                ],
                "size_um": [
                    float(cavity['width'] / self.unit),
                    float(cavity['height'] / self.unit)
                ],
                "frequency_hz": 1.485e6 * cavity['zero'] / cavities[0]['zero']
            })
        
        # Write metadata file
        metadata_file = gds_file.replace('.gds', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_file}")
        
        # Generate layer map for GDS viewers
        self.generate_layer_map(gds_file)
    
    def generate_layer_map(self, gds_file):
        """Generate layer map file for GDS viewers"""
        
        layer_map_file = gds_file.replace('.gds', '_layers.txt')
        
        with open(layer_map_file, 'w') as f:
            f.write("# Φ-24 Layer Map for GDS Viewers\n")
            f.write("# Layer : DataType : Description : Color : Pattern\n")
            f.write("# -------------------------------------------------\n")
            
            # Fibonacci layers
            for i in range(101, 122):
                f.write(f"{i}:0 : Fibonacci Layer {i-100} : #00FF00 : dot\n")
            
            # Other layers
            layer_info = [
                (200, 0, "SiO2 Isolation", "#C8C8C8", "solid"),
                (201, 0, "Nb Ground Plane", "#0000FF", "cross"),
                (300, 0, "JJ Base Electrode", "#FF0000", "solid"),
                (301, 1, "JJ Tunnel Barrier", "#FFFF00", "dot"),
                (302, 0, "JJ Counter Electrode", "#FF0000", "solid"),
                (303, 0, "JJ Shunt Resistor", "#FFA500", "solid"),
                (304, 0, "Passivation", "#800080", "hatch"),
                (400, 0, "Resonant Cavity", "#FF00FF", "hatch"),
                (401, 0, "Waveguide", "#00FFFF", "solid"),
                (402, 10, "RF Feedline", "#008080", "solid"),
                (500, 0, "Phase Control", "#FF4500", "solid"),
                (501, 0, "Temperature Sensor", "#8B4513", "solid"),
                (600, 0, "Test Structures", "#A52A2A", "dot"),
                (999, 0, "Seal Ring", "#000000", "solid"),
            ]
            
            for layer, datatype, desc, color, pattern in layer_info:
                f.write(f"{layer}:{datatype} : {desc} : {color} : {pattern}\n")
        
        print(f"Layer map saved to: {layer_map_file}")


def generate_phi24_core(output_dir='.'):
    """Generate the complete Φ-24 core layout"""
    
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize layout generator
    layout = Phi24Layout(unit=1e-9, precision=1e-12)
    
    # Generate layout
    output_file = os.path.join(output_dir, 'phi24_core.gds')
    gds_file = layout.generate_layout(output_file)
    
    print(f"\nΦ-24 layout generated successfully!")
    print(f"Main GDS file: {gds_file}")
    
    # Generate additional files
    generate_additional_files(output_dir)
    
    return gds_file


def generate_additional_files(output_dir):
    """Generate additional support files"""
    
    import os
    import json
    
    # 1. DRC rule file
    drc_rules = {
        "min_width": {
            "JJ_BASE": 0.1,
            "JJ_COUNTER": 0.1,
            "JJ_SHUNT": 0.2,
            "CONTROL": 1.0,
            "WAVEGUIDE": 10.0,
        },
        "min_spacing": {
            "JJ_BASE": 0.2,
            "JJ_COUNTER": 0.2,
            "JJ_BASE_to_JJ_COUNTER": 0.1,
            "CAVITY_to_CAVITY": 25.0,
            "CONTROL": 2.0,
        },
        "fibonacci_rules": {
            "thickness_ratio": 1.61803,
            "thickness_tolerance": 0.0005,
            "interface_roughness_max": 0.2,
        }
    }
    
    drc_file = os.path.join(output_dir, 'phi24_drc_rules.json')
    with open(drc_file, 'w') as f:
        json.dump(drc_rules, f, indent=2)
    
    # 2. Fabrication instructions
    fab_instructions = """Φ-24 Temporal Resonator Fabrication Instructions
==================================================================

1. SUBSTRATE PREPARATION
   - Material: c-plane Sapphire, 2" diameter
   - Thickness: 500 ± 25 μm
   - Surface preparation: RCA clean + 1000°C O₂ anneal

2. FIBONACCI SUPERLATTICE GROWTH
   - MBE system: Base pressure < 5×10⁻¹¹ Torr
   - Sequence: 21 layers following Fibonacci word F₈
   - Thickness control: 1.618 nm (Bi₂Se₃) / 1.000 nm (NbSe₂)
   - Growth monitoring: In-situ RHEED + QCM

3. JOSEPHSON JUNCTION FABRICATION
   - Base electrode: Nb, 200 nm, DC magnetron sputtering
   - Tunnel barrier: Al, 10 nm thermal evaporation + oxidation
   - Counter electrode: Nb, 200 nm
   - Shunt resistors: AuPd, 20 nm, lift-off process

4. RESONANT CAVITY DEFINITION
   - Cavity dimensions: 50 × 50 × 1.618 μm³
   - Isolation: 25 μm spacing between cavities
   - Coupling: < -30 dB target

5. CRYOGENIC PACKAGING
   - Temperature: 20 mK operating point
   - Magnetic shielding: Triple μ-metal + superconducting shield
   - RF filtering: π-filter at 1.485 MHz

CRITICAL PARAMETERS:
   - Fibonacci ratio: 1.61803 ± 0.0005
   - Resonance frequency: 1.485000 MHz ± 100 Hz
   - Temporal wedge: 11.00 ± 0.01 ns
   - P-ECC convergence: > 0.985
   - Josephson critical current: 150 μA ± 3 μA

VERIFICATION PROTOCOL:
   1. Room temperature: Structural verification (XRD, TEM, AFM)
   2. 4.2 K: Superconducting transition verification
   3. 20 mK: Riemann Lock verification
   4. P-ECC convergence test

SAFETY:
   - Cryogenic handling: Full face shield, cryo gloves
   - RF radiation: < 10 mW at sample
   - Magnetic fields: < 1 mG in sample area
"""
    
    fab_file = os.path.join(output_dir, 'phi24_fabrication_instructions.txt')
    with open(fab_file, 'w') as f:
        f.write(fab_instructions)
    
    # 3. Python script to load and analyze the GDS
    analysis_script = """#!/usr/bin/env python3
"""
    analysis_file = os.path.join(output_dir, 'analyze_phi24_layout.py')
    with open(analysis_file, 'w') as f:
        f.write(analysis_script)
    
    print(f"\nAdditional files generated in {output_dir}:")
    print(f"  - {os.path.basename(drc_file)}")
    print(f"  - {os.path.basename(fab_file)}")
    print(f"  - {os.path.basename(analysis_file)}")


def main():
    """Main function to generate Φ-24 layout"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Φ-24 GDSII layout')
    parser.add_argument('-o', '--output', default='.', 
                       help='Output directory (default: current directory)')
    parser.add_argument('-u', '--unit', type=float, default=1e-9,
                       help='Database unit in meters (default: 1e-9 = 1 nm)')
    parser.add_argument('-p', '--precision', type=float, default=1e-12,
                       help='Layout precision (default: 1e-12)')
    
    args = parser.parse_args()
    
    print("Φ-24 Temporal Resonator Layout Generator")
    print("=" * 50)
    
    try:
        # Generate layout
        gds_file = generate_phi24_core(args.output)
        
        print("\n" + "=" * 50)
        print("LAYOUT GENERATION COMPLETE")
        print("=" * 50)
        print("\nNext steps:")
        print("1. View layout in KLayout or similar GDS viewer")
        print("2. Run DRC checks against foundry rules")
        print("3. Generate mask data for fabrication")
        print("4. Begin MBE growth of Fibonacci superlattice")
        print("\nFor fabrication questions: foundry@ctt-science.org")
        
    except Exception as e:
        print(f"\nError generating layout: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
