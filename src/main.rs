use std::sync::Arc;

use vulkano::{device::{physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags}, format::Format, image::ImageUsage, instance::{Instance, InstanceCreateInfo}, swapchain::{ColorSpace, Surface, SurfaceInfo, Swapchain, SwapchainCreateInfo}, VulkanLibrary};
use winit::{dpi::LogicalSize, event::{Event, WindowEvent}, event_loop::EventLoop, window::WindowBuilder};



fn select_device(
    vlk_inst: &Arc<Instance>,
    vlk_surface: &Arc<Surface>
) -> (Arc<Device>, impl ExactSizeIterator + Iterator<Item = Arc<Queue>>) {
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
        .min_by_key(|(p, _)| {
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("no suitable physical device found");
    
    Device::new(physical_device, DeviceCreateInfo {
        queue_create_infos: vec![QueueCreateInfo {
            queue_family_index,
            ..Default::default()
        }],
        enabled_extensions: DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        },
        ..Default::default()
    }).unwrap()
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
            enabled_extensions: Surface::required_extensions(window.as_ref()),
            ..Default::default()
        }
    ).unwrap();
    let vlk_surface              = Surface::from_window(vlk_inst.clone(), window.clone()).unwrap();
    let (vlk_device, vlk_queues) = select_device(&vlk_inst, &vlk_surface);
    let vlk_queues: Vec<_>       = vlk_queues.collect();
    let vlk_surface_cap          = vlk_device.physical_device().surface_capabilities(&vlk_surface, SurfaceInfo::default()).unwrap();
    let vlk_chain                = Swapchain::new(vlk_device, vlk_surface.clone(), SwapchainCreateInfo {
        image_usage: ImageUsage::COLOR_ATTACHMENT,
        image_format: Format::B8G8R8A8_SRGB,
        image_extent: [window.inner_size().width, window.inner_size().height],
        image_color_space: ColorSpace::SrgbNonLinear,
        min_image_count: vlk_surface_cap.min_image_count,
        ..Default::default()
    }).unwrap();
    println!("Num queues: {}", vlk_queues.len());
    
    event_loop.run(move |event, _, control_flow| {
        control_flow.set_poll();
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                control_flow.set_exit();
            }
            Event::MainEventsCleared => {
                // does a draw her
            },
            _ => (),
        }
    });
}
