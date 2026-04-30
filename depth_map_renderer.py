# depth_map_renderer.py
from panda3d.core import *

class DepthMapRenderer:
    def __init__(self, base):
        self.base = base
        self.depth_texture = None
        self.overlay_node = None
        self.min_depth = 0.1
        self.max_depth = 100.0
        self.gradient_start = 0.2
        self.gradient_end = 0.4
        self.depth_buffer = None
        self.depth_camera_np = None
        self.setup_depth_render()
        self.setup_depth_overlay()

    def set_gradient_start(self, value):
        self.gradient_start = value
        if self.overlay_node:
            self.overlay_node.setShaderInput("gradientStart", value)
    
    def set_gradient_end(self, value):
        self.gradient_end = value
        if self.overlay_node:
            self.overlay_node.setShaderInput("gradientEnd", value)
    
    def setup_depth_render(self):
        win_width = 1920
        win_height = 1080
        
        self.depth_texture = Texture()
        self.depth_texture.setup_2d_texture(win_width, win_height, Texture.T_float, Texture.F_depth_component32)
        
        fb_props = FrameBufferProperties()
        fb_props.set_depth_bits(32)
        fb_props.set_float_depth(True)
        
        self.depth_buffer = self.base.win.make_texture_buffer("depth_buffer", win_width, win_height, 
                                                              self.depth_texture, to_ram=True)
        
        if self.depth_buffer is None:
            self.depth_buffer = self.base.graphicsEngine.make_output(
                self.base.pipe, "depth_buffer", 0, 
                fb_props,
                WindowProperties.size(win_width, win_height),
                GraphicsPipe.BF_refuse_window,
                self.base.win.get_gsg(), self.base.win
            )
            if self.depth_buffer:
                self.depth_buffer.add_render_texture(
                    self.depth_texture, 
                    GraphicsOutput.RTM_copy_ram, 
                    GraphicsOutput.RTP_depth
                )
        
        self.depth_buffer.set_clear_color_active(False)
        self.depth_buffer.set_clear_depth_active(True)
        self.depth_buffer.set_clear_depth(1.0)
        
        lens = PerspectiveLens()
        lens.set_near_far(self.min_depth, self.max_depth)
        
        main_lens = self.base.cam.node().get_lens()
        if hasattr(main_lens, 'get_fov'):
            main_fov = main_lens.get_fov()
            lens.set_fov(main_fov)
        else:
            lens.set_fov(60)
        
        depth_camera = Camera('depth_camera', lens)
        self.depth_camera_np = self.base.render.attach_new_node(depth_camera)
        
        depth_region = self.depth_buffer.make_display_region(0, 1, 0, 1)
        depth_region.set_camera(self.depth_camera_np)
        depth_region.set_clear_depth_active(True)
        depth_region.set_clear_depth(1.0)
        
        depth_region.set_clear_color_active(False)
            
    def setup_depth_overlay(self):
        win_width = self.base.win.getXSize()
        win_height = self.base.win.getYSize()

        cm = CardMaker('depth_overlay')
        cm.setFrame(-1, 1, -1, 1)

        self.overlay_node = self.base.render2d.attachNewNode(cm.generate())
        self.overlay_node.setPos(0, 0, 0)

        vertex_shader = """
        #version 330
        uniform mat4 p3d_ModelViewProjectionMatrix;
        in vec4 p3d_Vertex;
        in vec2 p3d_MultiTexCoord0;
        out vec2 texcoord;
        void main() {
            gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
            texcoord = p3d_MultiTexCoord0;
        }
        """

        fragment_shader = """
        #version 330
        uniform sampler2D depthMap;
        uniform float near;
        uniform float far;
        uniform float gradientStart;
        uniform float gradientEnd;
        in vec2 texcoord;
        out vec4 fragColor;

        float linearize_depth(float depth) {
            return (2.0 * near) / (far + near - depth * (far - near));
        }

        void main() {
            float depth = texture(depthMap, texcoord).r;
            float linear_depth = linearize_depth(depth);
            float normalized_depth = (linear_depth - gradientStart) / (gradientEnd - gradientStart);
            normalized_depth = clamp(normalized_depth, 0.0, 1.0);
            float t = 1.0 - normalized_depth;
            vec3 color;

            if (t >= 0.9) {
                float segment_t = (t - 0.9) / 0.1;
                color = mix(vec3(1.0, 0.0, 0.0), vec3(0.5, 0.0, 0.0), segment_t);
            } else if (t >= 0.7) {
                float segment_t = (t - 0.7) / 0.2;
                color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), segment_t);
            } else if (t >= 0.5) {
                float segment_t = (t - 0.5) / 0.2;
                color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), segment_t);
            } else if (t >= 0.3) {
                // Изумрудный -> жёлтый
                float segment_t = (t - 0.3) / 0.2;
                color = mix(vec3(0.1, 0.7, 0.4), vec3(1.0, 1.0, 0.0), segment_t);
            } else if (t >= 0.1) {
                // Голубой -> изумрудный
                float segment_t = (t - 0.1) / 0.2;
                color = mix(vec3(0.0, 0.0, 1.0), vec3(0.1, 0.7, 0.4), segment_t);
            } else {
                float segment_t = t / 0.1;
                // Тёмно-синий -> синий
                color = mix(vec3(0.0, 0.0, 0.3), vec3(0.0, 0.0, 1.0), segment_t);
            }

            fragColor = vec4(color, 1.0);
        }
        """

        shader = Shader.make(Shader.SL_GLSL, vertex_shader, fragment_shader)
        self.overlay_node.setShader(shader)
        depth_stage = TextureStage('depth_stage')
        depth_stage.setMode(TextureStage.M_modulate)
        self.overlay_node.setTexture(depth_stage, self.depth_texture)
        self.overlay_node.setShaderInput("depthMap", self.depth_texture)
        self.overlay_node.setShaderInput("near", self.min_depth)
        self.overlay_node.setShaderInput("far", self.max_depth)
        self.overlay_node.setShaderInput("gradientStart", self.gradient_start)
        self.overlay_node.setShaderInput("gradientEnd", self.gradient_end)
        self.overlay_node.setTransparency(TransparencyAttrib.MAlpha)
        self.overlay_node.setBin("fixed", 50)
        self.overlay_node.setDepthTest(False)
        self.overlay_node.setDepthWrite(False)
        self.overlay_node.hide()
    
    def update_depth_texture(self):
        if not hasattr(self, 'depth_buffer') or not self.depth_buffer:
            return False
        
        original_camera = self.base.camera
        
        try:
            main_cam_pos = original_camera.get_pos(self.base.render)
            main_cam_hpr = original_camera.get_hpr(self.base.render)
            
            self.depth_camera_np.set_pos(main_cam_pos)
            self.depth_camera_np.set_hpr(main_cam_hpr)
            
            if hasattr(original_camera.node(), 'get_lens'):
                main_lens = original_camera.node().get_lens()
                depth_lens = self.depth_camera_np.node().get_lens()
                
                if hasattr(main_lens, 'get_near') and hasattr(main_lens, 'get_far'):
                    self.min_depth = main_lens.get_near()
                    self.max_depth = main_lens.get_far()
                    depth_lens.set_near_far(self.min_depth, self.max_depth)
                
                if hasattr(main_lens, 'get_fov'):
                    fov = main_lens.get_fov()
                    depth_lens.set_fov(fov)
            
            self.base.camera = self.depth_camera_np
            
            self.depth_buffer.set_active(True)
            self.base.graphicsEngine.render_frame()
            
            if self.depth_texture:
                self.depth_texture.reload()
            
            if self.overlay_node and self.depth_texture:
                self.overlay_node.setShaderInput("depthMap", self.depth_texture)
                self.overlay_node.setShaderInput("near", self.min_depth)
                self.overlay_node.setShaderInput("far", self.max_depth)
                self.overlay_node.setShaderInput("gradientStart", self.gradient_start)
                self.overlay_node.setShaderInput("gradientEnd", self.gradient_end)
            
            return True
            
        finally:
            self.base.camera = original_camera
            self.depth_buffer.set_active(False)
    
    def toggle_overlay(self):
        if self.overlay_node:
            if self.overlay_node.isHidden():
                self.overlay_node.show()
            else:
                self.overlay_node.hide()
            return not self.overlay_node.isHidden()
        return False
    
    def set_overlay_visibility(self, visible):
        if self.overlay_node:
            if visible:
                self.overlay_node.show()
            else:
                self.overlay_node.hide()