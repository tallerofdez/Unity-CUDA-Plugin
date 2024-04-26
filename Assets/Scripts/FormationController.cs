using System.Collections;
using Unity.Collections; 
using Unity.Collections.LowLevel.Unsafe; 
using System.Collections.Generic;
using UnityEngine;
using System.Runtime.InteropServices;

public class FormationController : MonoBehaviour
{

    [SerializeField] private GameObject prefab;
    [SerializeField] public int numParticles;

    private List<GameObject> particles;

    private float cube;
    private float offset;
    
    private unsafe float* posX;
    private unsafe float* posY;
    private unsafe float* posZ;

    private unsafe float* initialPosX;
    private unsafe float* initialPosY;
    private unsafe float* initialPosZ;
    public Vector3 dir;
    public float period = 1;


    private const string pluginName = "CudaPlugin";
    
    [DllImport(pluginName)]
    private static extern unsafe void cubeFormation(float* posX, float* posY, float* posZ, int particles, int cube, float offset);


    [DllImport(pluginName)]
    private static extern unsafe void cubeMovement(float* posX, float* posY, float* posZ, float* initialPosX, float* initialPosY, float* initialPosZ, int particles, int cube, float ciclos);


    [DllImport(pluginName)]
    private static extern unsafe void InitialPos(float* posX, float* posY, float* posZ, float* initialPosX, float* initialPosY, float* initialPosZ, int particles, int cube);


    void Awake()
    {
        offset = prefab.transform.localScale.x;
        cube = Mathf.Pow(numParticles, 1f / 3f);
        numParticles = (int)cube * (int)cube *  (int)cube;
    }

    // Start is called before the first frame update
    void Start()
    { 
        
        particles = new List<GameObject>();

        unsafe
        {
            NativeArray<float> tempX = new NativeArray<float>(numParticles, Allocator.Temp);
            NativeArray<float> tempY = new NativeArray<float>(numParticles, Allocator.Temp);
            NativeArray<float> tempZ = new NativeArray<float>(numParticles, Allocator.Temp);


            NativeArray<float> initialTempX = new NativeArray<float>(numParticles, Allocator.Temp);
            NativeArray<float> initialTempY = new NativeArray<float>(numParticles, Allocator.Temp);
            NativeArray<float> initialTempZ = new NativeArray<float>(numParticles, Allocator.Temp);



            posX = (float*)tempX.GetUnsafePtr();
            posY = (float*)tempY.GetUnsafePtr();
            posZ = (float*)tempZ.GetUnsafePtr();
                

            initialPosX = (float*)initialTempX.GetUnsafePtr();
            initialPosY = (float*)initialTempY.GetUnsafePtr();
            initialPosZ = (float*)initialTempZ.GetUnsafePtr();

            
            tempX.Dispose();
            tempY.Dispose();
            tempZ.Dispose();

            initialTempX.Dispose();
            initialTempY.Dispose();
            initialTempZ.Dispose();
            

            cubeFormation(posX, posY, posZ, numParticles, (int)cube, offset);

            for (int i = 0; i < numParticles; i++)
            {

                //Debug.Log(posX[i] + "|" + posY[i] + "|" + posZ[i]);

                Vector3 targetPosition = new Vector3(posX[i], posY[i], posZ[i]);
                GameObject instance = Instantiate(prefab);
                instance.transform.position = targetPosition;
                particles.Add(instance);
            }

           InitialPos(posX, posY, posZ, initialPosX, initialPosY, initialPosZ, numParticles, (int)cube);

           
        }//end unsafe

    }

    void Update()
    {
        float ciclos = Time.time / period;
        
        unsafe
        {
          
           cubeMovement(posX, posY, posZ, initialPosX, initialPosY, initialPosZ, numParticles, (int)cube, ciclos);
        
           for (int i = 0; i < particles.Count;  i++) {
               particles[i].transform.position = new Vector3(posX[i], posY[i], posZ[i]);
               //Debug.Log(posX[i] + "|" + posY[i] + "|" + posZ[i]);
           }

            
        }
        


    }


    private void OnApplicationQuit()
    {
       //posX.Free();
    }

}