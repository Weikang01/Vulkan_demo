workspace "Vulkan_tutorial"
    architecture ("x86_64")

    startproject "tutorial"

    configurations
    {
        "Debug",
        "Release"
    }

outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"
vulkan_sdk = "E:/VulkanSDK/1.2.176.1"

-- Include directories relative to root folder (solution directory)
IncludeDir = {}
IncludeDir["GLFW"] = "library/GLFW/include"
IncludeDir["glm"] = "library/glm"
IncludeDir["stb"] = "library/stb"
IncludeDir["vulkan"] = vulkan_sdk .. "/include"

LibDir = {}
LibDir["vulkan"] = vulkan_sdk .. "/lib"

include "library/GLFW"

project "tutorial"
    location "tutorial"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"
    staticruntime "On"

    targetdir ("bin/".. outputdir .. "/%{prj.name}")
    objdir ("bin-int/".. outputdir .. "/%{prj.name}")

    files
    {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/Shaders/**.vert",
        "%{prj.name}/src/Shaders/**.frag"
    }

    includedirs
    {
        "%{prj.name}/src",
        "%{IncludeDir.vulkan}",
        "%{IncludeDir.GLFW}",
        "%{IncludeDir.glm}",
        "%{IncludeDir.stb}"
    }

    libdirs
    {
        "%{LibDir.vulkan}"
    }

    links
    {
        "GLFW",
        "vulkan-1"
    }

    filter "system:windows"
        staticruntime "On"
        systemversion "latest"

    filter "configurations:Debug"
        defines 
        {
            "_DEBUG",
            "_CONSOLE",
            "_LIB"
        }
        runtime "Debug"
        symbols "On"

    filter "configurations:Release"
        defines 
        {
            "NDEBUG",
            "_CONSOLE",
            "_LIB"
        }
        runtime "Release"
        optimize "On"
