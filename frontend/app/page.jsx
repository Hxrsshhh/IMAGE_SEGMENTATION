"use client"

import { useState } from "react"
import ImageUpload from "../components/ImageUpload"
import ResultView from "@/components/Resultview"

export default function Home(){

const [result,setResult] = useState(null)

return(

<div className="flex flex-col items-center p-10">

<h1 className="text-3xl font-bold mb-6">
Terrain Segmentation AI
</h1>

<ImageUpload setResult={setResult}/>

<ResultView result={result}/>

</div>

)

}