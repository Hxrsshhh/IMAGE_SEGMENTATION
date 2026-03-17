"use client"

import { useState } from "react"

export default function ImageUpload({setResult}){

const [file,setFile] = useState(null)
const [loading,setLoading] = useState(false)

const handleUpload = async () => {

if(!file) return

setLoading(true)

const formData = new FormData()
formData.append("file",file)

const res = await fetch("http://localhost:8000/predict",{
method:"POST",
body:formData
})

const data = await res.json()

setResult(data.result)

setLoading(false)

}

return(

<div className="flex flex-col gap-4">

<input
type="file"
onChange={(e)=>setFile(e.target.files[0])}
/>

<button
onClick={handleUpload}
className="bg-blue-500 text-white px-4 py-2 rounded"
>

{loading ? "Processing..." : "Run Segmentation"}

</button>

</div>

)

}