use smallvec::SmallVec;
use std::{collections::HashSet, ops::Range, sync::Arc, time::Duration};
use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        pool::{CommandPool, CommandPoolCreateFlags, CommandPoolCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBufferAbstract,
        RenderPassBeginInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::{ClearValue, Format},
    image::{
        sampler::{ComponentMapping, ComponentSwizzle},
        view::{ImageView, ImageViewCreateInfo, ImageViewType},
        ImageAspects, ImageLayout, ImageSubresourceRange, ImageUsage, SampleCount,
    },
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineSubpassType,
            vertex_input::VertexInputState,
            viewport::{Scissor, Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{
        AttachmentDescription, AttachmentLoadOp, AttachmentReference, AttachmentStoreOp,
        Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, Subpass,
        SubpassDescription,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    swapchain::{
        self, ColorSpace, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::GpuFuture,
    VulkanLibrary,
};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

fn load_shader(path: &str) -> Vec<u32> {
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
    buf
}

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

pub fn main() {
    let event_loop = EventLoop::new();
    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    window.set_title("GPU Ray Tracer");
    window.set_inner_size(LogicalSize::new(300, 300));

    let vlk_lib = VulkanLibrary::new().unwrap();
    let vlk_inst = Instance::new(
        vlk_lib,
        InstanceCreateInfo {
            enabled_extensions: InstanceExtensions {
                ext_debug_utils: true,
                ..Surface::required_extensions(window.as_ref())
            },
            ..Default::default()
        },
    )
    .unwrap();
    let vlk_surface = Surface::from_window(vlk_inst.clone(), window.clone()).unwrap();
    let (vlk_device, vlk_queues) = select_device(&vlk_inst, &vlk_surface);
    let vlk_queues: Vec<_> = vlk_queues.collect();
    let vlk_surface_cap = vlk_device
        .physical_device()
        .surface_capabilities(&vlk_surface, SurfaceInfo::default())
        .unwrap();
    let (vlk_chain, vlk_chain_imgs) = Swapchain::new(
        vlk_device.clone(),
        vlk_surface.clone(),
        SwapchainCreateInfo {
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            image_format: Format::B8G8R8A8_SRGB,
            image_extent: [window.inner_size().width, window.inner_size().height],
            image_color_space: ColorSpace::SrgbNonLinear,
            min_image_count: vlk_surface_cap.min_image_count,
            ..Default::default()
        },
    )
    .unwrap();

    let _callback = DebugUtilsMessenger::new(
        vlk_inst.clone(),
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

    let frag_spirv = load_shader("./shader/frag.spv");
    let vert_spirv = load_shader("./shader/vert.spv");

    let frag_shader = unsafe {
        ShaderModule::new(
            vlk_device.clone(),
            ShaderModuleCreateInfo::new(frag_spirv.as_slice()),
        )
    }
    .unwrap();
    let vert_shader = unsafe {
        ShaderModule::new(
            vlk_device.clone(),
            ShaderModuleCreateInfo::new(vert_spirv.as_slice()),
        )
    }
    .unwrap();

    let frag_shader_stage =
        PipelineShaderStageCreateInfo::new(frag_shader.entry_point("main").unwrap());
    let vert_shader_stage =
        PipelineShaderStageCreateInfo::new(vert_shader.entry_point("main").unwrap());
    let input_assembly = InputAssemblyState {
        topology: PrimitiveTopology::TriangleList,
        primitive_restart_enable: false,
        ..Default::default()
    };
    let dynamic_state = [DynamicState::Viewport, DynamicState::Scissor];
    let vertex_state = VertexInputState::new();
    let color_blend = ColorBlendAttachmentState {
        blend: None,
        ..Default::default()
    };
    let pipeline_layout =
        PipelineLayout::new(vlk_device.clone(), PipelineLayoutCreateInfo::default()).unwrap();
    let raster_state = RasterizationState {
        cull_mode: Default::default(),
        front_face: Default::default(),
        line_width: 1.0,
        ..Default::default()
    };
    let multisample_state = MultisampleState {
        rasterization_samples: SampleCount::Sample1,
        sample_shading: None,
        ..Default::default()
    };

    let render_pass = RenderPass::new(
        vlk_device.clone(),
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
    .unwrap();

    let pipeline = GraphicsPipeline::new(vlk_device.clone(), None, {
        let mut info = GraphicsPipelineCreateInfo::layout(pipeline_layout.clone());
        info.stages = SmallVec::from_vec(vec![vert_shader_stage, frag_shader_stage]);
        info.vertex_input_state = None;
        info.input_assembly_state = Some(input_assembly);
        info.viewport_state = Some(ViewportState::default());
        info.rasterization_state = Some(raster_state);
        info.multisample_state = Some(multisample_state);
        info.depth_stencil_state = None;
        info.color_blend_state = Some(ColorBlendState {
            attachments: vec![color_blend],
            ..Default::default()
        });
        info.dynamic_state = HashSet::from_iter(dynamic_state.iter().cloned());
        info.vertex_input_state = Some(vertex_state);
        info.layout = pipeline_layout;
        info.subpass = Some(PipelineSubpassType::BeginRenderPass(
            Subpass::from(render_pass.clone(), 0).unwrap(),
        ));
        info
    })
    .unwrap();

    let mut vlk_chain_fbs = vec![];
    for image in vlk_chain_imgs {
        vlk_chain_fbs.push(
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![ImageView::new(
                        image,
                        ImageViewCreateInfo {
                            view_type: ImageViewType::Dim2d,
                            format: vlk_chain.image_format(),
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
                    extent: vlk_chain.image_extent(),
                    ..Default::default()
                },
            )
            .unwrap(),
        );
    }
    let vlk_chain_fbs = vlk_chain_fbs;
    let cmd_alloc = StandardCommandBufferAllocator::new(
        vlk_device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 1,
            ..Default::default()
        },
    );

    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                // Get an image to render to from the swapchain.
                let next_img =
                    swapchain::acquire_next_image(vlk_chain.clone(), Some(Duration::from_secs(2)))
                        .unwrap();
                // next_img.2.wait(None).unwrap();
                let index = next_img.0 as usize;

                // Construct commands.
                let mut cmd_buf = AutoCommandBufferBuilder::primary(
                    &cmd_alloc,
                    vlk_queues[0].clone().queue_family_index(),
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();
                cmd_buf
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some(ClearValue::Float(
                                [0.0, 0.0, 0.0, 1.0].into(),
                            ))],
                            ..RenderPassBeginInfo::framebuffer(vlk_chain_fbs[index].clone())
                        },
                        Default::default(),
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    .set_viewport(
                        0,
                        SmallVec::from_vec(vec![Viewport {
                            offset: [0f32, 0f32],
                            extent: [
                                vlk_chain.image_extent()[0] as f32,
                                vlk_chain.image_extent()[1] as f32,
                            ],
                            ..Default::default()
                        }]),
                    )
                    .unwrap()
                    .set_scissor(
                        0,
                        SmallVec::from_vec(vec![Scissor {
                            offset: [0, 0],
                            extent: vlk_chain.image_extent(),
                        }]),
                    )
                    .unwrap()
                    .draw(3, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(Default::default())
                    .unwrap();
                let cmd_buf = cmd_buf.build().unwrap();

                // Run the commands.
                next_img
                    .2
                    .then_execute(vlk_queues[0].clone(), cmd_buf)
                    .unwrap()
                    .then_swapchain_present(
                        vlk_queues[0].clone(),
                        SwapchainPresentInfo::swapchain_image_index(
                            vlk_chain.clone(),
                            index as u32,
                        ),
                    )
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None)
                    .unwrap();
            }
            _ => (),
        }
    });
}
