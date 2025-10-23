import sys
import numpy as np
from pathlib import Path
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
                             QTabWidget, QTextEdit, QSplitter, QGroupBox, QScrollArea,
                             QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtGui import QIcon, QFont, QColor, QPalette
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GL import glGetString, GL_VERSION, GL_VENDOR, GL_RENDERER
from OpenGL.GL import shaders
import trimesh
import json

# Vertex Shader - runs on GPU for each vertex
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fragColor;
out vec3 fragNormal;
out vec3 fragPos;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    fragPos = vec3(model * vec4(position, 1.0));
    fragNormal = mat3(transpose(inverse(model))) * normal;
    fragColor = color;
}
"""

# Fragment Shader - runs on GPU for each pixel
FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
in vec3 fragNormal;
in vec3 fragPos;

out vec4 finalColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;
    
    vec3 result = (ambient + diffuse + specular) * fragColor;
    finalColor = vec4(result, 1.0);
}
"""

class GLBViewer(QOpenGLWidget):
    """GPU-accelerated OpenGL widget for rendering GLB models"""
    def __init__(self, parent=None):
        # Request OpenGL 3.3 Core Profile
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)  # 4x MSAA
        QSurfaceFormat.setDefaultFormat(fmt)
        
        super().__init__(parent)
        self.mesh = None
        self.rotation_x = 0
        self.rotation_y = 0
        self.zoom = -5.0
        self.last_pos = None
        self.animations = {}
        self.current_animation = None
        self.animation_time = 0.0
        
        # GPU buffers
        self.vao = None
        self.vbo_vertices = None
        self.vbo_normals = None
        self.vbo_colors = None
        self.ebo = None
        self.shader_program = None
        self.vertex_count = 0
        
    def initializeGL(self):
        v = glGetString(GL_VERSION)
        vendor = glGetString(GL_VENDOR)
        renderer = glGetString(GL_RENDERER)
        if v:
            print("OpenGL Version:", v.decode())
            print("OpenGL Vendor :", vendor.decode())
            print("OpenGL Renderer:", renderer.decode())
        else:
            print("⚠️ No GL context string available (context not current?)")
        """Initialize OpenGL with GPU shaders"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)  # Enable antialiasing
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glClearColor(0.15, 0.15, 0.2, 1.0)
        
        # Compile shaders on GPU
        try:
            vertex_shader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)
            print("✓ GPU shaders compiled successfully")
        except Exception as e:
            print(f"✗ Shader compilation error: {e}")
            
    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        if self.mesh is None or self.shader_program is None or self.vao is None:
            return
        
        if self.vao is not None and not glIsVertexArray(self.vao):
            self.upload_to_gpu()
            if not glIsVertexArray(self.vao):
                return
            
        glUseProgram(self.shader_program)
        
        # Create transformation matrices
        model = self.get_model_matrix()
        view = self.get_view_matrix()
        projection = self.get_projection_matrix()
        
        # Upload matrices to GPU
        model_loc = glGetUniformLocation(self.shader_program, "model")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, model)
        glUniformMatrix4fv(view_loc,  1, GL_TRUE, view)
        glUniformMatrix4fv(proj_loc,  1, GL_TRUE, projection)
        
        # Upload lighting parameters to GPU
        light_pos_loc = glGetUniformLocation(self.shader_program, "lightPos")
        view_pos_loc = glGetUniformLocation(self.shader_program, "viewPos")
        light_color_loc = glGetUniformLocation(self.shader_program, "lightColor")
        
        glUniform3f(light_pos_loc, 5.0, 5.0, 5.0)
        glUniform3f(view_pos_loc, 0.0, 0.0, -self.zoom)
        glUniform3f(light_color_loc, 1.0, 1.0, 1.0)
        
        # Draw using GPU
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.vertex_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
    def get_model_matrix(self):
        """Create model transformation matrix"""
        model = np.identity(4, dtype=np.float32)
        
        # Apply rotations
        angle_x = np.radians(self.rotation_x)
        angle_y = np.radians(self.rotation_y)
        
        # Rotation around X axis
        rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x), 0],
            [0, np.sin(angle_x), np.cos(angle_x), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation around Y axis
        ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        model = model @ rx @ ry
        return model
        
    def get_view_matrix(self):
        """Create view matrix"""
        view = np.identity(4, dtype=np.float32)
        view[2, 3] = self.zoom
        return view
        
    def get_projection_matrix(self):
        """Create perspective projection matrix"""
        fov = np.radians(45.0)
        aspect = self.width() / self.height() if self.height() != 0 else 1
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(fov / 2.0)
        projection = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
        
        return projection
        
    def load_glb(self, filepath):
        """Load GLB file and upload to GPU"""
        try:
            scene = trimesh.load(filepath)
            if isinstance(scene, trimesh.Scene):
                self.mesh = trimesh.util.concatenate([
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene.geometry.values()
                ])
            else:
                self.mesh = scene
                
            # Center and scale the mesh
            self.mesh.vertices -= self.mesh.centroid
            scale = 2.0 / self.mesh.extents.max()
            self.mesh.vertices *= scale
            
            # Compute normals for lighting
            self.mesh.vertex_normals
            
            # Upload mesh data to GPU
            self.upload_to_gpu()
            
            self.update()
            print(f"✓ Model loaded to GPU: {self.vertex_count} vertices")
            return True
        except Exception as e:
            print(f"✗ Error loading GLB: {e}")
            return False
            
    def upload_to_gpu(self):
        """Upload mesh data to GPU buffers (VBO/VAO)"""
        if self.mesh is None:
            return

        # Ensure this widget's GL context is current while touching GL state
        self.makeCurrent()
        try:
            # Clean up old buffers if they exist
            if self.vao is not None and glIsVertexArray(self.vao):
                glDeleteVertexArrays(1, [self.vao])
            if self.vbo_vertices is not None and glIsBuffer(self.vbo_vertices):
                glDeleteBuffers(1, [self.vbo_vertices])
            if self.vbo_normals is not None and glIsBuffer(self.vbo_normals):
                glDeleteBuffers(1, [self.vbo_normals])
            if self.vbo_colors is not None and glIsBuffer(self.vbo_colors):
                glDeleteBuffers(1, [self.vbo_colors])
            if self.ebo is not None and glIsBuffer(self.ebo):
                glDeleteBuffers(1, [self.ebo])

            # Prepare vertex data
            vertices = self.mesh.vertices.astype(np.float32)
            normals = self.mesh.vertex_normals.astype(np.float32)

            if hasattr(self.mesh.visual, 'vertex_colors') and self.mesh.visual.vertex_colors is not None:
                colors = (self.mesh.visual.vertex_colors[:, :3] / 255.0).astype(np.float32)
            else:
                colors = np.tile([0.7, 0.7, 0.8], (len(vertices), 1)).astype(np.float32)

            indices = self.mesh.faces.flatten().astype(np.uint32)
            self.vertex_count = len(indices)

            # Create VAO + buffers with current context
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)

            # VBO for vertices
            self.vbo_vertices = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(0)

            # VBO for normals
            self.vbo_normals = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_normals)
            glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(1)

            # VBO for colors
            self.vbo_colors = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
            glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(2)

            # EBO for indices
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

            glBindVertexArray(0)
            print(f"✓ Uploaded to GPU: {len(vertices)} vertices, {len(indices)} indices")
        except Exception as e:
            print(f"✗ An error occurred during GPU upload: {e}")
        finally:
            self.doneCurrent()
        
    def mousePressEvent(self, event):
        self.last_pos = event.pos()
        
    def mouseMoveEvent(self, event):
        if self.last_pos:
            dx = event.pos().x() - self.last_pos.x()
            dy = event.pos().y() - self.last_pos.y()
            
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.rotation_x += dy * 0.5
                self.rotation_y += dx * 0.5
                self.update()
                
            self.last_pos = event.pos()
            
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.zoom += delta / 120.0 * 0.5
        self.zoom = max(-20, min(-1, self.zoom))
        self.update()
        
    def animate(self, bone_name, property_name, value):
        """Apply animation to bones"""
        if self.mesh:
            if bone_name not in self.animations:
                self.animations[bone_name] = {}
            self.animations[bone_name][property_name] = value
            
            # Apply rotation animations
            if 'rotation' in property_name:
                if property_name == 'rotation_x':
                    self.rotation_x = value
                elif property_name == 'rotation_y':
                    self.rotation_y = value
                    
            self.update()


class VTubeStudioClone(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VTube Studio Clone - GPU Accelerated")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setup_style()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - 3D Viewer
        self.viewer = GLBViewer()
        splitter.addWidget(self.viewer)
        
        # Right panel - Controls
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.animation_speed = 1.0
        
    def setup_style(self):
        """Setup dark theme similar to VTube Studio"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', Arial;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 8px 15px;
                color: #cdd6f4;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45475a;
                border: 1px solid #6c7086;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
            QGroupBox {
                border: 2px solid #45475a;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 5px;
            }
            QSlider::groove:horizontal {
                background: #313244;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #89b4fa;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QTextEdit, QListWidget {
                background-color: #181825;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #313244;
                border: 1px solid #45475a;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #45475a;
                color: #89b4fa;
            }
            QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
    def create_right_panel(self):
        """Create the right control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Model controls
        model_group = QGroupBox("🎮 GPU-Accelerated Model Controls")
        model_layout = QVBoxLayout()
        
        load_btn = QPushButton("📁 Load GLB Model")
        load_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_btn)
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_layout.addWidget(self.model_label)
        
        # GPU info
        self.gpu_info = QLabel("GPU Ready")
        self.gpu_info.setStyleSheet("color: #a6e3a1; font-size: 10px;")
        self.gpu_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_layout.addWidget(self.gpu_info)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Tabs for different features
        tabs = QTabWidget()
        
        # Animation tab
        animation_tab = self.create_animation_tab()
        tabs.addTab(animation_tab, "🎬 Animation")
        
        # Code tab
        code_tab = self.create_code_tab()
        tabs.addTab(code_tab, "💻 Code")
        
        # Parameters tab
        params_tab = self.create_parameters_tab()
        tabs.addTab(params_tab, "⚙️ Parameters")
        
        layout.addWidget(tabs)
        
        return panel
        
    def create_animation_tab(self):
        """Create animation controls tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Animation controls
        controls_group = QGroupBox("Playback Controls")
        controls_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        play_btn = QPushButton("▶ Play")
        play_btn.clicked.connect(self.play_animation)
        pause_btn = QPushButton("⏸ Pause")
        pause_btn.clicked.connect(self.pause_animation)
        stop_btn = QPushButton("⏹ Stop")
        stop_btn.clicked.connect(self.stop_animation)
        
        btn_layout.addWidget(play_btn)
        btn_layout.addWidget(pause_btn)
        btn_layout.addWidget(stop_btn)
        controls_layout.addLayout(btn_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        speed_slider = QSlider(Qt.Orientation.Horizontal)
        speed_slider.setMinimum(1)
        speed_slider.setMaximum(300)
        speed_slider.setValue(100)
        speed_slider.valueChanged.connect(self.change_speed)
        speed_layout.addWidget(speed_slider)
        self.speed_label = QLabel("1.0x")
        speed_layout.addWidget(self.speed_label)
        controls_layout.addLayout(speed_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Bone list
        bone_group = QGroupBox("Bones")
        bone_layout = QVBoxLayout()
        self.bone_list = QListWidget()
        bone_layout.addWidget(self.bone_list)
        bone_group.setLayout(bone_layout)
        layout.addWidget(bone_group)
        
        layout.addStretch()
        return tab
        
    def create_code_tab(self):
        """Create code editor tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        info_label = QLabel("Write Python code to animate your model (GPU accelerated):")
        layout.addWidget(info_label)
        
        self.code_editor = QTextEdit()
        self.code_editor.setPlaceholderText(
            "# Example code:\n"
            "# animate('model', 'rotation_x', 15.0)\n"
            "# animate('model', 'rotation_y', -30.0)\n\n"
            "# Available functions:\n"
            "# animate(bone_name, property, value)\n"
            "# Properties: rotation_x, rotation_y, rotation_z\n"
            "#            position_x, position_y, position_z\n\n"
            "# Example animation loop:\n"
            "# for i in range(360):\n"
            "#     animate('model', 'rotation_y', i)\n"
        )
        layout.addWidget(self.code_editor)
        
        btn_layout = QHBoxLayout()
        run_btn = QPushButton("▶ Run Code")
        run_btn.clicked.connect(self.run_code)
        clear_btn = QPushButton("🗑 Clear")
        clear_btn.clicked.connect(self.code_editor.clear)
        
        btn_layout.addWidget(run_btn)
        btn_layout.addWidget(clear_btn)
        layout.addLayout(btn_layout)
        
        self.code_output = QTextEdit()
        self.code_output.setReadOnly(True)
        self.code_output.setMaximumHeight(100)
        self.code_output.setPlaceholderText("Output will appear here...")
        layout.addWidget(QLabel("Output:"))
        layout.addWidget(self.code_output)
        
        return tab
        
    def create_parameters_tab(self):
        """Create parameters control tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Expression parameters
        expr_group = QGroupBox("Expression Parameters")
        expr_layout = QVBoxLayout()
        
        parameters = [
            ("Mouth Open", 0, 100),
            ("Mouth Form", -100, 100),
            ("Eye Open Left", 0, 100),
            ("Eye Open Right", 0, 100),
            ("Brow Height", -100, 100),
        ]
        
        for param_name, min_val, max_val in parameters:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(param_name))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(0)
            param_layout.addWidget(slider)
            value_label = QLabel("0")
            param_layout.addWidget(value_label)
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
            expr_layout.addLayout(param_layout)
            
        expr_group.setLayout(expr_layout)
        scroll_layout.addWidget(expr_group)
        
        # Transform parameters
        transform_group = QGroupBox("Transform")
        transform_layout = QVBoxLayout()
        
        transforms = [
            ("Position X", -100, 100),
            ("Position Y", -100, 100),
            ("Rotation", -180, 180),
            ("Scale", 10, 200),
        ]
        
        for param_name, min_val, max_val in transforms:
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(param_name))
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(0 if "Position" in param_name or "Rotation" in param_name else 100)
            param_layout.addWidget(slider)
            value_label = QLabel("0")
            param_layout.addWidget(value_label)
            slider.valueChanged.connect(lambda v, l=value_label: l.setText(str(v)))
            transform_layout.addLayout(param_layout)
            
        transform_group.setLayout(transform_layout)
        scroll_layout.addWidget(transform_group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return tab
        
    def load_model(self):
        """Load GLB model file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load GLB Model", "", "GLB Files (*.glb *.gltf);;All Files (*)"
        )
        if filepath:
            if self.viewer.load_glb(filepath):
                self.model_label.setText(f"Loaded: {Path(filepath).name}")
                self.gpu_info.setText(f"✓ GPU: {self.viewer.vertex_count} vertices uploaded")
                self.code_output.append(f"✓ Model loaded to GPU: {filepath}")
            else:
                self.model_label.setText("Failed to load model")
                self.gpu_info.setText("✗ GPU: Upload failed")
                self.code_output.append(f"✗ Failed to load model")
                
    def play_animation(self):
        """Start animation playback"""
        self.timer.start(16)  # ~60 FPS
        self.code_output.append("▶ Animation playing (GPU accelerated)")
        
    def pause_animation(self):
        """Pause animation"""
        self.timer.stop()
        self.code_output.append("⏸ Animation paused")
        
    def stop_animation(self):
        """Stop animation"""
        self.timer.stop()
        self.viewer.animation_time = 0.0
        self.code_output.append("⏹ Animation stopped")
        
    def change_speed(self, value):
        """Change animation speed"""
        self.animation_speed = value / 100.0
        self.speed_label.setText(f"{self.animation_speed:.1f}x")
        
    def update_animation(self):
        """Update animation frame"""
        self.viewer.animation_time += 0.016 * self.animation_speed
        self.viewer.update()
        
    def run_code(self):
        """Execute user code"""
        code = self.code_editor.toPlainText()
        self.code_output.clear()
        
        # Create animation context
        def animate(bone_name, property_name, value):
            self.viewer.animate(bone_name, property_name, value)
            self.code_output.append(f"GPU: Animated {bone_name}.{property_name} = {value}")
            
        try:
            # Execute code in controlled environment
            exec(code, {"animate": animate, "np": np})
            self.code_output.append("\n✓ Code executed successfully on GPU")
        except Exception as e:
            self.code_output.append(f"\n✗ Error: {str(e)}")


def main():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setSwapInterval(1)

    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    app.setApplicationName("VTube Studio Clone - GPU")
    
    window = VTubeStudioClone()
    window.show()
    
    # Auto-load model - specify your model path here
    MODEL_PATH = r"C:\path\to\your\model.glb"  # Change this to your model path
    
    # Uncomment the lines below to auto-load a model on startup
    # import os
    # if os.path.exists(MODEL_PATH):
    #     window.viewer.load_glb(MODEL_PATH)
    #     window.model_label.setText(f"Loaded: {Path(MODEL_PATH).name}")
    #     window.gpu_info.setText(f"✓ GPU: {window.viewer.vertex_count} vertices uploaded")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()