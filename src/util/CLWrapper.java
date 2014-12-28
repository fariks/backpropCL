package util;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import memobjects.MemObject;
import org.jocl.*;

import java.io.File;
import java.io.IOException;
import java.util.Map;

import static org.jocl.CL.*;

public class CLWrapper {

    private static cl_context context = null;
    private static cl_command_queue commandQueue = null;
    private static cl_program program = null;

    /**
     * Private constructor to prevent instantiation
     */
    private CLWrapper() {
    }

    public static void cl_init(String programSource) {
        // The platform, device type and device number
        // that will be used
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_GPU;
        final int deviceIndex = 0;

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true);

        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];

        // Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);

        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];

        // Obtain a device ID
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];

        // Create a context for the selected device
        context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null);

        // Create a command-queue for the selected device
        commandQueue =
                clCreateCommandQueue(context, device, 0, null);

        // Create the program from the source code
        program = clCreateProgramWithSource(context,
                1, new String[]{loadProgram(programSource)}, null, null);

        // Build the program
        clBuildProgram(program, 0, null, null, null, null);
    }

    public static void cl_release(Map<MemObject, cl_mem> memObjects, cl_kernel... kernels) {
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);
        clReleaseMemObjects(memObjects);
        clReleaseKernels(kernels);
    }

    private static void clReleaseMemObjects(Map<MemObject, cl_mem> memObjects) {
        for (cl_mem object : memObjects.values()) {
            clReleaseMemObject(object);
        }
    }

    private static void clReleaseKernels(cl_kernel... kernels) {
        for (cl_kernel k : kernels) {
            clReleaseKernel(k);
        }
    }

    private static String loadProgram(String programSource) {
        try {
            return Files.toString(new File(programSource), Charsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public static cl_mem createBuffer(long flags, long size, float[] values) {
        return clCreateBuffer(context, flags, size, Pointer.to(values), null);
    }

    public static cl_mem createBuffer(long flags, long size, int[] values) {
        return clCreateBuffer(context, flags, size, Pointer.to(values), null);
    }

    public static cl_kernel createKernel(String name) {
        return clCreateKernel(program, name, null);
    }

    public static int runKernel(cl_kernel kernel, int work_dim, long[] global_work_size, long[] local_work_size) {
        return clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, null, global_work_size, local_work_size, 0, null, null);
    }

    public static int readBuffer(cl_mem buffer, long cb, float[] values) {
        return clEnqueueReadBuffer(commandQueue, buffer, CL_TRUE, 0, cb, Pointer.to(values), 0, null, null);
    }

    public static int readBuffer(cl_mem buffer, long cb, int[] values) {
        return clEnqueueReadBuffer(commandQueue, buffer, CL_TRUE, 0, cb, Pointer.to(values), 0, null, null);
    }

    public static int setKernelArg(cl_kernel kernel, int arg_index, long arg_size, cl_mem value)
    {
        return clSetKernelArg(kernel, arg_index, arg_size, Pointer.to(value));
    }

    public static int setKernelArg(cl_kernel kernel, int arg_index, long arg_size, int[] value)
    {
        return clSetKernelArg(kernel, arg_index, arg_size, Pointer.to(value));
    }

    public static int setKernelArg(cl_kernel kernel, int arg_index, long arg_size, float[] value)
    {
        return clSetKernelArg(kernel, arg_index, arg_size, Pointer.to(value));
    }

    public static int writeBuffer(cl_mem buffer, long cb, float[] values)
    {
        return clEnqueueWriteBuffer(commandQueue, buffer, CL_TRUE, 0, cb, Pointer.to(values), 0, null, null);
    }
}
