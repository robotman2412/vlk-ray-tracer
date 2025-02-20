mod scene;
mod shader_buffer;

use glam::{Mat4, Quat, Vec3};
use scene::*;
use shader_buffer::{GpuScene, GpuSkybox};
use smallvec::SmallVec;
use std::{collections::HashSet, f32::consts::PI, ops::Range, process::Command, sync::Arc};
use vulkano::{
    buffer::{
        view::{BufferView, BufferViewCreateInfo},
        *,
    },
    command_buffer::{allocator::*, *},
    descriptor_set::{allocator::*, *},
    device::{physical::*, *},
    format::*,
    image::{sampler::*, view::*, *},
    instance::{debug::*, *},
    memory::allocator::*,
    pipeline::{
        compute::*,
        graphics::{
            color_blend::*, input_assembly::*, multisample::*, rasterization::*, subpass::*,
            vertex_input::*, viewport::*, *,
        },
        layout::*,
        *,
    },
    render_pass::*,
    shader::*,
    swapchain::{self, *},
    sync::GpuFuture,
    VulkanLibrary,
};
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalSize, Size},
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowAttributes},
};

/// Struct that holds all contexts needed for the Vulkan API.
struct Context {
    instance: Arc<Instance>,
    device: Arc<Device>,
    surface: Arc<Surface>,
    queues: Vec<Arc<Queue>>,
    allocator: Arc<StandardMemoryAllocator>,
    swapchain: Option<Arc<Swapchain>>,
    swapchain_images: Vec<Arc<Image>>,
    swapchain_framebuffers: Vec<Arc<Framebuffer>>,
    render_pass: Option<Arc<RenderPass>>,
    gfx_pipeline: Option<Arc<GraphicsPipeline>>,
    rt_pipeline: Option<Arc<ComputePipeline>>,
    rt_samples: Option<Arc<Image>>,
    desc_alloc: Option<Arc<StandardDescriptorSetAllocator>>,
    cmd_alloc: Option<Arc<StandardCommandBufferAllocator>>,
}

/// Struct of push constants for the fragment shader.
#[repr(C)]
#[derive(Debug, Copy, Clone, BufferContents)]
struct FragParams {
    frame_counter: u32,
}

/// Push constants for the ray tracer.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
struct RtPushConst {
    params: RtParams,
    object_count: u32,
}

/// Parameters for the ray tracer.
#[repr(C)]
#[derive(Copy, Clone, BufferContents)]
struct RtParams {
    /// Matrix representing camera position and orientation.
    cam_matrix: [f32; 16],
    /// Tangent of half the vertical field of view.
    cam_v_fov: f32,
    /// Current frame counter starting at 1.
    frame_counter: u32,
}

/// Load a SPIR-V shader from a file.
fn load_shader(
    device: Arc<Device>,
    path: &str,
) -> Result<Arc<ShaderModule>, vulkano::Validated<vulkano::VulkanError>> {
    use std::fs::File;
    use std::io::Read;
    let mut file = File::open(path).unwrap();
    let mut buf = Vec::<u32>::new();
    loop {
        let mut tmp = [0u8; 4];
        let length = file.read(&mut tmp).unwrap();
        if length == 0 {
            break;
        } else if length % 4 != 0 {
            panic!("Invalid SPIR-V file");
        }
        buf.push(u32::from_le_bytes(tmp));
    }
    unsafe { ShaderModule::new(device, ShaderModuleCreateInfo::new(buf.as_slice())) }
}

/// Select the most suitable physical device and queues.
fn select_device(
    vlk_inst: &Arc<Instance>,
    vlk_surface: &Arc<Surface>,
) -> (
    Arc<Device>,
    impl ExactSizeIterator + Iterator<Item = Arc<Queue>>,
) {
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = vlk_inst
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| {
            // Require the extensions.
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            // Get compatible queue types.
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // Drawing to a window requires the graphics flag.
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, vlk_surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .expect("no suitable physical device found");

    Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: DeviceExtensions {
                khr_swapchain: true,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
}

/// Create framebuffers from swapchain images.
fn create_swapchain_fbs(ctx: &mut Context) -> Vec<Arc<Framebuffer>> {
    ctx.swapchain_images
        .iter()
        .map(|image| {
            Framebuffer::new(
                ctx.render_pass.clone().unwrap(),
                FramebufferCreateInfo {
                    attachments: vec![ImageView::new(
                        image.clone(),
                        ImageViewCreateInfo {
                            view_type: ImageViewType::Dim2d,
                            format: ctx.swapchain.clone().unwrap().image_format(),
                            component_mapping: ComponentMapping {
                                r: ComponentSwizzle::Identity,
                                g: ComponentSwizzle::Identity,
                                b: ComponentSwizzle::Identity,
                                a: ComponentSwizzle::Identity,
                            },
                            subresource_range: ImageSubresourceRange {
                                aspects: ImageAspects::COLOR,
                                mip_levels: Range { start: 0, end: 1 },
                                array_layers: Range { start: 0, end: 1 },
                            },
                            ..Default::default()
                        },
                    )
                    .unwrap()],
                    layers: 1,
                    extent: ctx.swapchain.clone().unwrap().image_extent(),
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}

/// Create a (new) swapchain and its framebuffers.
/// Default size is 400x300.
fn create_swapchain(ctx: &mut Context, window_size: [u32; 2]) {
    let surface_cap = ctx
        .device
        .physical_device()
        .surface_capabilities(&ctx.surface, SurfaceInfo::default())
        .unwrap();

    let (swapchain, images) = Swapchain::new(
        ctx.device.clone(),
        ctx.surface.clone(),
        SwapchainCreateInfo {
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            image_format: Format::B8G8R8A8_SRGB,
            image_extent: window_size,
            image_color_space: ColorSpace::SrgbNonLinear,
            min_image_count: surface_cap.min_image_count,
            ..Default::default()
        },
    )
    .unwrap();
    ctx.swapchain = Some(swapchain);
    ctx.swapchain_images = images;
    ctx.swapchain_framebuffers = create_swapchain_fbs(ctx);
}

/// Recreate the swapchain for a new resolution.
fn recreate_swapchain(ctx: &mut Context, window_size: [u32; 2]) {
    let surface_cap = ctx
        .device
        .physical_device()
        .surface_capabilities(&ctx.surface, SurfaceInfo::default())
        .unwrap();

    let (swapchain, images) = ctx
        .swapchain
        .as_mut()
        .unwrap()
        .recreate(SwapchainCreateInfo {
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            image_format: Format::B8G8R8A8_SRGB,
            image_extent: window_size,
            image_color_space: ColorSpace::SrgbNonLinear,
            min_image_count: surface_cap.min_image_count,
            ..Default::default()
        })
        .unwrap();

    ctx.swapchain = Some(swapchain);
    ctx.swapchain_images = images;
    ctx.swapchain_framebuffers = create_swapchain_fbs(ctx);
}

/// Draw a single frame.
fn draw(ctx: &mut Context, frame_counter: u32) {
    // Get an image to render to from the swapchain.
    let next_img = swapchain::acquire_next_image(ctx.swapchain.clone().unwrap(), None).unwrap();
    let index = next_img.0 as usize;

    // Create image attachment for the graphics pipeline to display the ray-traced image.
    let desc_set = DescriptorSet::new(
        ctx.desc_alloc.clone().unwrap(),
        ctx.gfx_pipeline.as_ref().unwrap().layout().set_layouts()[0].clone(),
        [WriteDescriptorSet::image_view(
            0,
            ImageView::new_default(ctx.rt_samples.clone().unwrap()).unwrap(),
        )],
        [],
    )
    .unwrap();

    // Construct commands.
    let mut cmd_buf = AutoCommandBufferBuilder::primary(
        ctx.cmd_alloc.clone().unwrap(),
        ctx.queues[0].clone().queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();
    cmd_buf
        .begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some(ClearValue::Float([0.0, 0.0, 0.0, 1.0].into()))],
                ..RenderPassBeginInfo::framebuffer(ctx.swapchain_framebuffers[index].clone())
            },
            Default::default(),
        )
        .unwrap()
        .bind_pipeline_graphics(ctx.gfx_pipeline.clone().unwrap())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            ctx.gfx_pipeline.as_ref().unwrap().layout().clone(),
            0,
            desc_set,
        )
        .unwrap()
        .push_constants(
            ctx.gfx_pipeline.as_ref().unwrap().layout().clone(),
            0,
            FragParams { frame_counter },
        )
        .unwrap()
        .set_viewport(
            0,
            SmallVec::from_vec(vec![Viewport {
                offset: [0f32, 0f32],
                extent: [
                    ctx.swapchain.clone().unwrap().image_extent()[0] as f32,
                    ctx.swapchain.clone().unwrap().image_extent()[1] as f32,
                ],
                ..Default::default()
            }]),
        )
        .unwrap()
        .set_scissor(
            0,
            SmallVec::from_vec(vec![Scissor {
                offset: [0, 0],
                extent: ctx.swapchain.clone().unwrap().image_extent(),
            }]),
        )
        .unwrap();
    unsafe { cmd_buf.draw(3, 1, 0, 0) }
        .unwrap()
        .end_render_pass(Default::default())
        .unwrap();
    let cmd_buf = cmd_buf.build().unwrap();

    // Run the commands.
    next_img
        .2
        .then_execute(ctx.queues[0].clone(), cmd_buf)
        .unwrap()
        .then_swapchain_present(
            ctx.queues[0].clone(),
            SwapchainPresentInfo::swapchain_image_index(
                ctx.swapchain.clone().unwrap(),
                index as u32,
            ),
        )
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

/// Create the ray-tracing samples buffer.
fn create_rt_samples(ctx: &mut Context, extent: [u32; 2]) {
    ctx.rt_samples = Some(
        Image::new(
            ctx.allocator.clone(),
            ImageCreateInfo {
                extent: [extent[0], extent[1], 1],
                format: Format::R32G32B32A32_SFLOAT,
                usage: ImageUsage::STORAGE | ImageUsage::INPUT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .unwrap(),
    );
}

/// Tell the GPU to collect a single ray-trace sample.
fn raytrace(ctx: &mut Context, params: &RtParams, scene: &GpuScene) {
    let mut cmd_buf = AutoCommandBufferBuilder::primary(
        ctx.cmd_alloc.clone().unwrap(),
        ctx.queues[0].clone().queue_family_index(),
        CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    let desc_set = DescriptorSet::new(
        ctx.desc_alloc.clone().unwrap(),
        ctx.rt_pipeline.as_ref().unwrap().layout().set_layouts()[0].clone(),
        [
            WriteDescriptorSet::image_view(
                0,
                ImageView::new_default(ctx.rt_samples.clone().unwrap()).unwrap(),
            ),
            WriteDescriptorSet::buffer(1, scene.objects.clone()),
            WriteDescriptorSet::buffer(2, scene.skybox.clone()),
        ],
        [],
    )
    .unwrap();

    cmd_buf
        .bind_pipeline_compute(ctx.rt_pipeline.clone().unwrap())
        .unwrap()
        .push_constants(
            ctx.rt_pipeline.as_ref().unwrap().layout().clone(),
            0,
            RtPushConst {
                params: *params,
                object_count: scene.object_count,
            },
        )
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            ctx.rt_pipeline.as_ref().unwrap().layout().clone(),
            0,
            desc_set,
        )
        .unwrap();

    // The shader must run once per pixel; it is grouped into 8x8 tiles.
    let groups = [
        (ctx.swapchain.clone().unwrap().image_extent()[0] + 7) / 8,
        (ctx.swapchain.clone().unwrap().image_extent()[1] + 7) / 8,
        1,
    ];
    unsafe { cmd_buf.dispatch(groups) }.unwrap();
    let cmd_buf = cmd_buf.build().unwrap();

    // Run the commands.
    cmd_buf
        .execute(ctx.queues[0].clone())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
}

struct App {
    ctx: Option<Context>,
    window: Option<Arc<Window>>,
    cpu_scene: Scene,
    gpu_scene: Option<GpuScene>,
    rt_params: RtParams,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("GPU Ray Tracer")
                        .with_inner_size(Size::Physical(PhysicalSize::new(400, 300))),
                )
                .unwrap(),
        );
        self.window = Some(window.clone());

        let vlk_lib = VulkanLibrary::new().unwrap();
        let vlk_inst = Instance::new(
            vlk_lib,
            InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    ext_debug_utils: true,
                    ..Surface::required_extensions(window.as_ref()).unwrap()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let vlk_surface = Surface::from_window(vlk_inst.clone(), window.clone()).unwrap();
        let (vlk_device, vlk_queues) = select_device(&vlk_inst, &vlk_surface);

        let mut ctx = Context {
            instance: vlk_inst,
            device: vlk_device.clone(),
            surface: vlk_surface,
            queues: vlk_queues.collect(),
            allocator: Arc::new(GenericMemoryAllocator::new_default(vlk_device)),
            swapchain: None,
            swapchain_images: vec![],
            swapchain_framebuffers: vec![],
            render_pass: None,
            gfx_pipeline: None,
            rt_pipeline: None,
            rt_samples: None,
            desc_alloc: None,
            cmd_alloc: None,
        };

        let _callback = DebugUtilsMessenger::new(
            ctx.instance.clone(),
            DebugUtilsMessengerCreateInfo::user_callback(unsafe {
                DebugUtilsMessengerCallback::new(|severity, _, callback_data| {
                    let severity = match severity {
                        DebugUtilsMessageSeverity::ERROR => "error",
                        DebugUtilsMessageSeverity::WARNING => "warning",
                        DebugUtilsMessageSeverity::INFO => "info",
                        DebugUtilsMessageSeverity::VERBOSE => "verbose",
                        _ => panic!(),
                    };
                    println!("[{}] {:?}", severity, callback_data.message);
                })
            }),
        )
        .unwrap();

        let frag_shader = load_shader(ctx.device.clone(), "./shader/frag.spv").unwrap();
        let vert_shader = load_shader(ctx.device.clone(), "./shader/vert.spv").unwrap();
        let rt_shader = load_shader(ctx.device.clone(), "./shader/rt.spv").unwrap();

        let dynamic_state = [DynamicState::Viewport, DynamicState::Scissor];
        let frag_shader_stage =
            PipelineShaderStageCreateInfo::new(frag_shader.entry_point("main").unwrap());
        let vert_shader_stage =
            PipelineShaderStageCreateInfo::new(vert_shader.entry_point("main").unwrap());
        let gfx_pipeline_layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&frag_shader_stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();

        ctx.render_pass = Some(
            RenderPass::new(
                ctx.device.clone(),
                RenderPassCreateInfo {
                    attachments: vec![AttachmentDescription {
                        format: Format::B8G8R8A8_SRGB,
                        samples: SampleCount::Sample1,
                        load_op: AttachmentLoadOp::Clear,
                        store_op: AttachmentStoreOp::Store,
                        initial_layout: ImageLayout::Undefined,
                        final_layout: ImageLayout::PresentSrc,
                        ..Default::default()
                    }],
                    subpasses: vec![SubpassDescription {
                        color_attachments: vec![Some(AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        })],
                        ..Default::default()
                    }],
                    dependencies: vec![],
                    ..Default::default()
                },
            )
            .unwrap(),
        );

        ctx.gfx_pipeline = Some(
            GraphicsPipeline::new(ctx.device.clone(), None, {
                let mut info = GraphicsPipelineCreateInfo::layout(gfx_pipeline_layout.clone());
                info.stages = SmallVec::from_vec(vec![frag_shader_stage, vert_shader_stage]);
                info.vertex_input_state = None;
                info.input_assembly_state = Some(InputAssemblyState {
                    topology: PrimitiveTopology::TriangleList,
                    primitive_restart_enable: false,
                    ..Default::default()
                });
                info.viewport_state = Some(ViewportState::default());
                info.rasterization_state = Some(RasterizationState {
                    cull_mode: Default::default(),
                    front_face: Default::default(),
                    line_width: 1.0,
                    ..Default::default()
                });
                info.multisample_state = Some(MultisampleState {
                    rasterization_samples: SampleCount::Sample1,
                    sample_shading: None,
                    ..Default::default()
                });
                info.depth_stencil_state = None;
                info.color_blend_state = Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState::default()],
                    ..Default::default()
                });
                info.dynamic_state = HashSet::from_iter(dynamic_state.iter().cloned());
                info.vertex_input_state = Some(VertexInputState::new());
                info.layout = gfx_pipeline_layout;
                info.subpass = Some(PipelineSubpassType::BeginRenderPass(
                    Subpass::from(ctx.render_pass.clone().unwrap(), 0).unwrap(),
                ));
                info
            })
            .unwrap(),
        );

        // The ray-tracing pipeline uses the following:
        // - A linear buffer of RGB floats that accumulates samples
        // - The inverse camera matrix
        // - The camera matrix
        // - The camera resolution as two ints
        let rt_shader_stage =
            PipelineShaderStageCreateInfo::new(rt_shader.entry_point("main").unwrap());
        let rt_pipeline_layout = PipelineLayout::new(
            ctx.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&rt_shader_stage])
                .into_pipeline_layout_create_info(ctx.device.clone())
                .unwrap(),
        )
        .unwrap();
        ctx.desc_alloc = Some(Arc::new(StandardDescriptorSetAllocator::new(
            ctx.device.clone(),
            Default::default(),
        )));
        ctx.rt_pipeline = Some(
            ComputePipeline::new(
                ctx.device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(rt_shader_stage, rt_pipeline_layout),
            )
            .unwrap(),
        );

        ctx.cmd_alloc = Some(Arc::new(StandardCommandBufferAllocator::new(
            ctx.device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        )));

        // Set everything up for the first frame.
        let window_size = Into::<[u32; 2]>::into(window.inner_size());
        create_swapchain(&mut ctx, window_size);
        create_rt_samples(&mut ctx, window_size);
        self.gpu_scene = Some(GpuScene::build(ctx.allocator.clone(), &self.cpu_scene).unwrap());

        self.ctx = Some(ctx);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if self
                    .ctx
                    .as_ref()
                    .unwrap()
                    .swapchain
                    .as_ref()
                    .unwrap()
                    .image_extent()
                    != Into::<[u32; 2]>::into(self.window.as_ref().unwrap().inner_size())
                {
                    let window_size =
                        Into::<[u32; 2]>::into(self.window.as_ref().unwrap().inner_size());
                    recreate_swapchain(self.ctx.as_mut().unwrap(), window_size);
                    create_rt_samples(self.ctx.as_mut().unwrap(), window_size);
                    self.rt_params.frame_counter = 0;
                }
                self.rt_params.frame_counter = 1;
                raytrace(
                    self.ctx.as_mut().unwrap(),
                    &self.rt_params,
                    &self.gpu_scene.as_ref().unwrap(),
                );
                draw(self.ctx.as_mut().unwrap(), self.rt_params.frame_counter);
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

pub fn main() {
    if !Command::new("glslc")
        .args(&["shader/vert.vert", "-o", "shader/vert.spv"])
        .status()
        .expect("Can't run glslc")
        .success()
    {
        panic!("Failed to compile vertex shader");
    }
    if !Command::new("glslc")
        .args(&["shader/frag.frag", "-o", "shader/frag.spv"])
        .status()
        .expect("Can't run glslc")
        .success()
    {
        panic!("Failed to compile fragment shader");
    }
    if !Command::new("glslc")
        .args(&[
            "-fshader-stage=comp",
            "-std=450core",
            "shader/rt.glsl",
            "-o",
            "shader/rt.spv",
        ])
        .status()
        .expect("Can't run glslc")
        .success()
    {
        panic!("Failed to compile ray tracing shader");
    }
    println!("Shaders compiled successfully");

    let scene = Scene {
        objects: vec![
            Box::new(Sphere {
                transform: Transform::from(Mat4::from_scale_rotation_translation(
                    Vec3::new(0.0, 0.0, 2.0),
                    Quat::IDENTITY,
                    Vec3::splat(0.5),
                )),
                prop: PhysProp::from_color(Vec3::new(1.0, 0.0, 0.0)),
            }),
            Box::new(Sphere {
                transform: Transform::from(Mat4::from_scale_rotation_translation(
                    Vec3::new(-1.0, 0.0, 2.0),
                    Quat::IDENTITY,
                    Vec3::splat(0.4),
                )),
                prop: PhysProp {
                    color: Vec3::new(0.0, 1.0, 0.0),
                    opacity: 1.0,
                    ior: 1.0,
                    roughness: 0.0,
                    emission: Vec3::ZERO,
                },
            }),
            Box::new(Plane {
                transform: Transform::from(Mat4::from_rotation_translation(
                    Quat::from_rotation_x(PI * 0.5),
                    Vec3::new(0.0, 0.5, 2.0),
                )),
                prop: PhysProp::from_color(Vec3::new(0.5, 0.5, 0.5)),
            }),
            Box::new(Sphere {
                transform: Transform::from(Mat4::from_scale_rotation_translation(
                    Vec3::new(-0.5, 0.3, 1.5),
                    Quat::IDENTITY,
                    Vec3::splat(0.2),
                )),
                prop: PhysProp::from_emission(Vec3::new(1.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0)),
            }),
            Box::new(Sphere {
                transform: Transform::from(Mat4::from_scale_rotation_translation(
                    Vec3::new(-0.3, 0.1, 1.2),
                    Quat::IDENTITY,
                    Vec3::splat(0.15),
                )),
                prop: PhysProp {
                    ior: 1.5,
                    opacity: 0.0,
                    roughness: 1.0,
                    color: Vec3::new(1.0, 1.0, 1.0),
                    emission: Vec3::ZERO,
                },
            }),
        ],
        skybox: Default::default(),
    };

    let mut app = App {
        ctx: None,
        window: None,
        rt_params: RtParams {
            cam_matrix: Mat4::IDENTITY.to_cols_array(),
            cam_v_fov: 1.0,
            frame_counter: 0,
        },
        cpu_scene: scene,
        gpu_scene: None,
    };
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop.run_app(&mut app).unwrap();
}
