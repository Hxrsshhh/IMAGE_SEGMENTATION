export default function ResultView({result}){

if(!result) return null

return(

<div className="mt-6">

<h2 className="text-xl font-bold">
Segmentation Result
</h2>

<img
src={`http://localhost:8000/${result}`}
className="mt-4 border"
/>

</div>

)

}